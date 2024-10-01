import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from monai.utils import set_determinism

class Trainer:
    def __init__(
        self,
        env_type,
        max_epochs,
        batch_size,
        device="cuda",
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
        self.ddp = self.env_type == "ddp"
        self.num_gpus = num_gpus
        self.device = device
        self.rank = 0
        self.local_rank = 0
        self.batch_size = batch_size
        self.logdir = logdir
        self.scheduler = None
        self.model = None
        self.auto_optim = True
        self.global_step = 0

        if self.ddp:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        torch.backends.cudnn.benchmark = True

        # Initialize SummaryWriter only on rank 0
        if self.rank == 0:
            os.makedirs(self.logdir, exist_ok=True)
            self.writer = SummaryWriter(self.logdir)
        else:
            self.writer = None

    def get_dataloader(self, dataset, shuffle=False, batch_size=1, train=True):
        if dataset is None:
            return None
        if self.ddp:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=4,
                sampler=sampler,
                pin_memory=True,
                worker_init_fn=self.worker_init_fn,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=4,
                pin_memory=True,
                worker_init_fn=self.worker_init_fn,
            )

    @staticmethod
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def prepare_batch(self, batch):
        if isinstance(batch, dict):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        elif isinstance(batch, list):
            batch = [item.to(self.device, non_blocking=True) for item in batch]
        elif isinstance(batch, torch.Tensor):
            batch = batch.to(self.device, non_blocking=True)
        else:
            raise TypeError("Unsupported batch type")
        return batch

    def distributed_concat(self, tensor, num_total_examples):
        output_tensors = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)[:num_total_examples]
        return concat

    def aggregate_val_outputs(self, val_outputs, return_list=False):
        val_outputs = torch.tensor(val_outputs).to(self.device)
        if self.ddp:
            dist.all_reduce(val_outputs, op=dist.ReduceOp.SUM)
            val_outputs = val_outputs / self.world_size

        if not return_list:
            mean_value = val_outputs.mean().item()
            return mean_value
        else:
            mean_values = val_outputs.mean(dim=0).cpu().tolist()
            return mean_values

    def train(
        self,
        train_dataset,
        optimizer=None,
        model=None,
        val_dataset=None,
        scheduler=None,
    ):
        set_determinism(1234)
        if model is not None:
            self.model = model

        if optimizer is not None:
            self.optimizer = optimizer

        if scheduler is not None:
            self.scheduler = scheduler

        train_loader = self.get_dataloader(
            train_dataset, shuffle=True, batch_size=self.batch_size, train=True
        )
        val_loader = (
            self.get_dataloader(val_dataset, shuffle=False, batch_size=1, train=False)
            if val_dataset
            else None
        )

        for epoch in range(self.max_epochs):
            self.epoch = epoch
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)

            self.train_epoch(train_loader, epoch)

            if (epoch + 1) % self.val_every == 0 and val_loader is not None:
                if self.model is not None:
                    self.model.eval()

                val_outputs = []
                for batch in val_loader:
                    batch = self.prepare_batch(batch)
                    with torch.no_grad():
                        val_out = self.validation_step(batch)
                    val_outputs.append(val_out)

                if self.rank == 0:
                    mean_val_outputs = self.aggregate_val_outputs(
                        val_outputs, return_list=isinstance(val_out, (list, tuple))
                    )
                    self.validation_end(mean_val_outputs=mean_val_outputs)

            if self.scheduler is not None:
                self.scheduler.step()

            if self.model is not None:
                self.model.train()

    def train_epoch(self, loader, epoch):
        if self.rank == 0:
            loader = tqdm(loader, total=len(loader), desc=f"Epoch {epoch}")

        for batch in loader:
            self.global_step += 1
            batch = self.prepare_batch(batch)

            loss = self.training_step(batch)

            if self.auto_optim:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.rank == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                loader.set_postfix(loss=loss.item(), lr=lr)

    def training_step(self, batch):
        raise NotImplementedError

    def validation_step(self, batch):
        raise NotImplementedError

    def validation_end(self, mean_val_outputs):
        pass

    def log(self, key, value, step):
        if self.rank == 0 and self.writer is not None:
            self.writer.add_scalar(key, value, step)

    def load_state_dict(self, weight_path, strict=True):
        state_dict = torch.load(weight_path, map_location="cpu")
        if "module" in state_dict:
            state_dict = state_dict["module"]
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "") if "module." in k else k
            new_state_dict[new_key] = v

        self.model.load_state_dict(new_state_dict, strict=strict)
        print(f"Model parameters loaded successfully.")
