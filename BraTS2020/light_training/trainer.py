# Trainer.py

import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from monai.inferers import SlidingWindowInferer
from monai.losses.dice import DiceLoss
from monai.utils import set_determinism

from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_new_model_and_delete_last
from light_training.evaluation.metric import dice


class Trainer:
    def __init__(self, env_type,
                 max_epochs,
                 batch_size,
                 device="cpu",
                 val_every=1,
                 num_gpus=1,
                 logdir="./logs/",
                 master_ip='localhost',
                 master_port=17750,
                 training_script="train.py",
                 ):
        assert env_type.lower() in ["pytorch", "ddp"], f"Unsupported env_type: {env_type}"
        self.env_type = env_type.lower()
        self.val_every = val_every
        self.max_epochs = max_epochs
        self.ddp = False
        self.num_gpus = num_gpus
        self.device = device
        self.rank = 0
        self.local_rank = 0
        self.batch_size = batch_size
        self.logdir = logdir
        self.scheduler = None 
        self.model = None
        self.auto_optim = True

        torch.backends.cudnn.enabled = True

        gpu_count = torch.cuda.device_count()
        if self.env_type == "ddp":
            self.ddp = True
            self.get_dist_args()
            self.initialize_distributed()
        else:
            if num_gpus > gpu_count:
                print("Number of GPUs requested exceeds available GPUs.")
                os._exit(0)

    def initialize_distributed(self):
        """Initialize torch.distributed."""
        if self.env_type != 'ddp':
            print('No need to initialize distributed training.')
            return

        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://'
        )
        self.world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()

        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device)

        print(f"Initialized distributed training on rank {self.rank}, device {self.device}, world size {self.world_size}")

    def get_dataloader(self, dataset, shuffle=False, batch_size=1, train=True):
        if dataset is None:
            return None
        if self.env_type == 'pytorch':
            return DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=12)
        else:
            if not train:
                sampler = DistributedSampler(dataset, shuffle=False)
            else:
                sampler = DistributedSampler(dataset, shuffle=True)
            return DataLoader(dataset,
                              batch_size=batch_size,
                              num_workers=12, 
                              sampler=sampler, 
                              drop_last=False)

    def get_dist_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--local_rank', type=int, default=0, help="local_rank")
        ds_args, _ = parser.parse_known_args()
        self.local_rank = ds_args.local_rank
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")

    def prepare_batch(self, batch):
        if isinstance(batch, dict):
            batch = {x: batch[x].contiguous().to(self.device) for x in batch if isinstance(batch[x], torch.Tensor)}
        elif isinstance(batch, list):
            batch = [x.to(self.device) for x in batch if isinstance(x, torch.Tensor)]
        elif isinstance(batch, torch.Tensor):
            batch = batch.to(self.device)
        else:
            print("Unsupported data type in batch")
            exit(0)
        return batch

    def aggregate_val_outputs(self, val_outputs, return_list=False):
        if not return_list:
            valid = ~torch.isnan(val_outputs)
            if valid.sum() == 0:
                return 0.0
            v_sum = torch.sum(val_outputs * valid.float()) / valid.sum()
            return v_sum.item()
        else:
            num_val = val_outputs.shape[1]
            v_sum = torch.zeros(num_val).to(self.device)
            length = torch.zeros(num_val).to(self.device)
            for i in range(num_val):
                valid = ~torch.isnan(val_outputs[:, i])
                v_sum[i] += torch.sum(val_outputs[valid, i])
                length[i] += torch.sum(valid)
            mean_val = torch.where(length > 0, v_sum / length, torch.zeros_like(v_sum))
            return mean_val.cpu().tolist()

    def distributed_concat(self, tensor, num_total_examples):
        output_tensors = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)[:num_total_examples]
        return concat

    def train(self,
              train_dataset,
              optimizer=None,
              model=None,
              val_dataset=None,
              scheduler=None,
             ):
        if scheduler is not None:
            self.scheduler = scheduler

        set_determinism(1234 + self.local_rank)
        if model is not None:
            print(f"Check model parameter: {next(model.parameters()).sum()}")
            para = sum([np.prod(list(p.size())) for p in model.parameters()])
            if self.rank == 0:
                print(f"Model parameters: {para * 4 / 1e6}M ")
            self.model = model

        self.global_step = 0

        if self.env_type == "pytorch":
            if self.model is not None:
                self.model.to(self.device)
            os.makedirs(self.logdir, exist_ok=True)
            self.writer = SummaryWriter(self.logdir)

        elif self.env_type == "ddp":
            if self.rank == 0:
                os.makedirs(self.logdir, exist_ok=True)
                self.writer = SummaryWriter(self.logdir)
            else:
                self.writer = None
            if self.model is not None:
                self.model.to(self.device)
                self.model = DDP(
                    self.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=True
                )

        else:
            print("Unsupported env_type")
            exit(0)

        train_loader = self.get_dataloader(train_dataset, shuffle=True, batch_size=self.batch_size, train=True)
        val_loader = self.get_dataloader(val_dataset, shuffle=False, batch_size=1, train=False) if val_dataset else None

        for epoch in range(0, self.max_epochs):
            self.epoch = epoch 
            if self.env_type == "ddp":
                train_loader.sampler.set_epoch(epoch)
                torch.distributed.barrier()
            self.train_epoch(train_loader, epoch)
            
            if (epoch + 1) % self.val_every == 0 and val_loader is not None:
                if self.model is not None:
                    self.model.eval()
                if self.env_type == "ddp":
                    torch.distributed.barrier()
                
                val_outputs = []
                for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
                    batch = self.prepare_batch(batch)
                    with torch.no_grad():
                        val_out = self.validation_step(batch)
                        assert val_out is not None 
                    val_outputs.append(val_out)
                
                if self.env_type == "ddp":
                    val_outputs = torch.tensor(val_outputs).to(self.device)
                    torch.distributed.barrier()
                    val_outputs = self.distributed_concat(val_outputs, num_total_examples=len(val_loader.sampler.dataset))
                else:
                    val_outputs = torch.tensor(val_outputs)
                
                if self.rank == 0:
                    mean_val_outputs = self.aggregate_val_outputs(val_outputs, return_list=isinstance(val_out, (list, tuple)))
                    self.validation_end(mean_val_outputs=mean_val_outputs)
            
            if self.scheduler is not None:
                self.scheduler.step()
            if self.model is not None:
                self.model.train()
        
        if self.env_type == "ddp":
            dist.destroy_process_group()

    def train_epoch(self, loader, epoch):
        if self.model is not None:
            self.model.train()
        if self.rank == 0:
            with tqdm(total=len(loader)) as t:
                for idx, batch in enumerate(loader):
                    self.global_step += 1
                    t.set_description(f'Epoch {epoch}')
                    batch = self.prepare_batch(batch)
                    
                    if self.model is not None:
                        for param in self.model.parameters():
                            param.grad = None
                    loss = self.training_step(batch)

                    if self.auto_optim:
                        loss.backward()
                        self.optimizer.step()
                        
                        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                        t.set_postfix(loss=loss.item(), lr=lr)
                    t.update(1)
        else:
            for idx, batch in enumerate(loader):
                self.global_step += 1
                batch = self.prepare_batch(batch)
                
                if self.model is not None:
                    for param in self.model.parameters():
                        param.grad = None

                loss = self.training_step(batch)
                if self.auto_optim:
                    loss.backward()
                    self.optimizer.step()

    def training_step(self, batch):
        raise NotImplementedError
    
    def validation_step(self, batch):
        raise NotImplementedError

    def validation_end(self, mean_val_outputs):
        pass 

    def log(self, k, v, step):
        if self.env_type == "pytorch":
            self.writer.add_scalar(k, scalar_value=v, global_step=step)
        elif self.env_type == "ddp":
            if self.rank == 0:
                self.writer.add_scalar(k, scalar_value=v, global_step=step)

    def load_state_dict(self, weight_path, strict=True):
        sd = torch.load(weight_path, map_location="cpu")
        if "module" in sd:
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module") else k 
            new_sd[new_k] = v 

        self.model.load_state_dict(new_sd, strict=strict)
        
        print(f"Model parameters loaded successfully.")


class BraTSTrainer(Trainer):
    def __init__(self, args):
        super().__init__(
            env_type=args.env_type,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            val_every=args.val_every,
            num_gpus=args.num_gpus,
            logdir=args.logdir,
            master_ip=args.master_ip,
            master_port=args.master_port,
            training_script=args.training_script
        )
        self.window_infer = SlidingWindowInferer(
            roi_size=[96, 96, 96],
            sw_batch_size=1,
            overlap=0.25
        )
        self.model = DiffUNet(number_modality=4, number_targets=3)

        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-3)
        self.ce = nn.CrossEntropyLoss() 
        self.mse = nn.MSELoss()
        self.scheduler = LinearWarmupCosineAnnealingLR(
            self.optimizer,
            warmup_epochs=30,
            max_epochs=args.max_epochs
        )

        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)

    def training_step(self, batch):
        image, label = self.get_input(batch)
        x_start = label * 2 - 1
        x_t, t, noise = self.model(x=x_start, pred_type="q_sample")
        pred_xstart = self.model(x=x_t, step=t, image=image, pred_type="denoise")

        loss_dice = self.dice_loss(pred_xstart, label)
        loss_bce = self.bce(pred_xstart, label)
        pred_xstart = torch.sigmoid(pred_xstart)
        loss_mse = self.mse(pred_xstart, label)

        loss = loss_dice + loss_bce + loss_mse

        if self.env_type == "ddp":
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss / dist.get_world_size()

        self.log("train_loss", loss.item(), step=self.global_step)

        return loss

    def get_input(self, batch):
        image = batch["image"]
        label = batch["label"]
        label = label.float()
        return image, label 

    def validation_step(self, batch):
        image, label = self.get_input(batch)    
        model = self.model.module if self.env_type == "ddp" else self.model
        output = self.window_infer(image, model, pred_type="ddim_sample")
        output = torch.sigmoid(output)
        output = (output > 0.5).float().cpu().numpy()

        target = label.cpu().numpy()
        wt = dice(output[:, 1], target[:, 1])  # CE
        tc = dice(output[:, 0], target[:, 0])  # Core
        et = dice(output[:, 2], target[:, 2])  # Active

        return [wt, tc, et]

    def validation_end(self, mean_val_outputs):
        wt, tc, et = mean_val_outputs
        self.log("wt", wt, step=self.epoch)
        self.log("tc", tc, step=self.epoch)
        self.log("et", et, step=self.epoch)

        mean_dice = (wt + tc + et) / 3
        self.log("mean_dice", mean_dice, step=self.epoch)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(
                self.model.module if self.env_type == "ddp" else self.model, 
                os.path.join(self.logdir, "model", f"best_model_{mean_dice:.4f}.pt"), 
                delete_symbol="best_model"
            )

        save_new_model_and_delete_last(
            self.model.module if self.env_type == "ddp" else self.model, 
            os.path.join(self.logdir, "model", f"final_model_{mean_dice:.4f}.pt"), 
            delete_symbol="final_model"
        )

        print(f"wt: {wt}, tc: {tc}, et: {et}, mean_dice: {mean_dice}")
