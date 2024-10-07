import numpy as np
from dataset.brats_data_utils_multi_label import get_loader_brats
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_new_model_and_delete_last
from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder
import argparse
from monai.losses.dice import DiceLoss
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
import yaml
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
from tqdm import tqdm
from tqdm import tqdm

# Import the updated Trainer class
from light_training.trainer import Trainer

set_determinism(1234)

number_modality = 4
number_targets = 3  # WT, TC, ET

class DiffUNet(nn.Module):
    def __init__(self, number_modality, number_targets):
        super().__init__()
        self.embed_model = BasicUNetEncoder(3, number_modality, number_targets, [64, 64, 128, 256, 512, 64])

        self.model = BasicUNetDe(3, number_targets, number_targets, [64, 64, 128, 256, 512, 64],
                                 act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))

        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(1000, [1000]),
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,
        )

        self.sample_diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(1000, [50]),
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,
        )
        self.sampler = UniformSampler(1000)

    def forward(self, image=None, x=None, pred_type=None, step=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, _ = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embeddings=embeddings)

    def ddim_inference(self, image):
        with torch.no_grad():
            embeddings = self.embed_model(image)
            sample_out = self.sample_diffusion.ddim_sample_loop(
                self.model,
                (1, number_targets, 96, 96, 96),
                model_kwargs={"image": image, "embeddings": embeddings},
            )
            return sample_out["pred_xstart"]

class BraTSTrainer(Trainer):
    def __init__(self, args):
        super().__init__(
            env_type=args.env,
            max_epochs=args.max_epoch,
            batch_size=args.batch_size,
            device=args.device,
            val_every=args.val_every,
            num_gpus=args.num_gpus,
            logdir=args.logdir,
            master_ip='localhost',
            master_port=17751,
            training_script=__file__,
        )

        self.window_infer = SlidingWindowInferer(
            roi_size=[96, 96, 96],
            sw_batch_size=1,
            overlap=0.25,
        )
        self.best_mean_dice = 0.0

        # Initialize the model
        self.model = DiffUNet(number_modality=4, number_targets=3)

        # Move model to device before wrapping with DDP
        self.model.to(self.device)

        # Wrap the model with DDP if using distributed training
        if self.ddp:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,  # Set to False to reduce overhead if there are no unused parameters
            )

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-3)
        self.scheduler = LinearWarmupCosineAnnealingLR(
            self.optimizer, warmup_epochs=30, max_epochs=args.max_epoch
        )

        self.dice_loss = DiceLoss(sigmoid=True)
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def get_input(self, batch):
        image = batch["image"].to(self.device)
        label = batch["label"].float().to(self.device)
        return image, label

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

        # Detach loss for reduction
        reduced_loss = loss.detach()
        if self.ddp:
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
            reduced_loss = reduced_loss / dist.get_world_size()

        if self.rank == 0:
            self.log("train_loss", reduced_loss.item(), step=self.global_step)

        return loss

    def validation_step(self, batch):
        #print("Starting validation step")
        image, label = self.get_input(batch)
        #print("Got input batch")

        # Ensure model is in evaluation mode
        self.model.eval()
        #print("Model set to eval mode")

        with torch.no_grad():
            print("Starting sliding window inference")
            if self.ddp:
                output = self.window_infer(
                    inputs=image,
                    network=lambda img: self.model.module.ddim_inference(img),
                )
            else:
                output = self.window_infer(
                    inputs=image,
                    network=lambda img: self.model.ddim_inference(img),
                )
            #print("Sliding window inference completed")

            output = torch.sigmoid(output)
            output = (output > 0.5).float().cpu().numpy()
            target = label.cpu().numpy()

            wt = dice(output[:, 1], target[:, 1])  # Whole tumor
            tc = dice(output[:, 0], target[:, 0])  # Tumor core
            et = dice(output[:, 2], target[:, 2])  # Enhancing tumor

            # Aggregate metrics across GPUs
            wt_tensor = torch.tensor(wt, device=self.device)
            tc_tensor = torch.tensor(tc, device=self.device)
            et_tensor = torch.tensor(et, device=self.device)

            if self.ddp:
                dist.all_reduce(wt_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(tc_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(et_tensor, op=dist.ReduceOp.SUM)

                world_size = dist.get_world_size()
                wt = wt_tensor.item() / world_size
                tc = tc_tensor.item() / world_size
                et = et_tensor.item() / world_size

            #print(f"Validation step completed: wt={wt}, tc={tc}, et={et}")
            return [wt, tc, et]

    def validate(self, val_dataset):
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        val_outputs = []
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            val_outputs.append(self.validation_step(batch))

        mean_val_outputs = np.mean(val_outputs, axis=0).tolist()
        self.validation_end(mean_val_outputs)

    def validation_end(self, mean_val_outputs):
        wt, tc, et = mean_val_outputs

        mean_dice = (wt + tc + et) / 3

        if self.rank == 0:
            self.log("wt", wt, step=self.epoch)
            self.log("tc", tc, step=self.epoch)
            self.log("et", et, step=self.epoch)
            self.log("mean_dice", mean_dice, step=self.epoch)

            if mean_dice > self.best_mean_dice:
                self.best_mean_dice = mean_dice
                save_new_model_and_delete_last(
                    self.model.module if self.ddp else self.model,
                    os.path.join(self.logdir, "model", f"best_model_{mean_dice:.4f}.pt"),
                    delete_symbol="best_model",
                )

            save_new_model_and_delete_last(
                self.model.module if self.ddp else self.model,
                os.path.join(self.logdir, "model", f"final_model_{mean_dice:.4f}.pt"),
                delete_symbol="final_model",
            )

            print(f"wt: {wt}, tc: {tc}, et: {et}, mean_dice: {mean_dice}")

    def validate(self, val_dataset):
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        val_outputs = []
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            val_outputs.append(self.validation_step(batch))

        mean_val_outputs = np.mean(val_outputs, axis=0)
        self.validation_end(mean_val_outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DiffUNet model for BraTS dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--logdir", type=str, default="./logs/", help="Directory to save logs and models")
    parser.add_argument("--max_epoch", type=int, default=300, help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--val_every", type=int, default=10, help="Validation frequency (in epochs)")
    parser.add_argument("--num_gpus", type=int, default=2, help="Number of GPUs to use")
    parser.add_argument("--train_size", type=int, default=None, help="Number of training samples for debugging")
    parser.add_argument("--val_size", type=int, default=None, help="Number of validation samples for debugging")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    parser.add_argument("--env", type=str, default="ddp", choices=["pytorch", "ddp"], help="Environment type")
    
    args = parser.parse_args()

    # Get local rank from environment variable
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set up distributed training if using DDP
    if args.env == "ddp":
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

    # Create the trainer
    trainer = BraTSTrainer(args)

    # Prepare data loaders
    train_ds, val_ds, test_ds = get_loader_brats(data_dir=args.data_dir, batch_size=args.batch_size, fold=0)

    # Reduce training and validation dataset sizes for debugging if specified
    if args.train_size is not None:
        train_ds = torch.utils.data.Subset(train_ds, range(args.train_size))
    if args.val_size is not None:
        val_ds = torch.utils.data.Subset(val_ds, range(args.val_size))
    print(f"Train dataset size: {len(train_ds)}, Validation dataset size: {len(val_ds)}")

    # Start training
    trainer.train(train_dataset=train_ds, val_dataset=val_ds)

    # Clean up distributed training resources
    if args.env == "ddp":
        dist.destroy_process_group()