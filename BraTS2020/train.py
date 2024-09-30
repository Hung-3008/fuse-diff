# train.py

import argparse
import os
import torch
import torch.nn as nn
from monai.utils import set_determinism

from dataset.brats_data_utils_multi_label import get_loader_brats
from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder
from guided_diffusion.gaussian_diffusion import (
    get_named_beta_schedule,
    ModelMeanType,
    ModelVarType,
    LossType
)
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_new_model_and_delete_last
from monai.inferers import SlidingWindowInferer
from monai.losses.dice import DiceLoss
from light_training.evaluation.metric import dice

from Trainer import BraTSTrainer

# Set determinism for reproducibility
set_determinism(123)

number_modality = 4
number_targets = 3  # WT, TC, ET

class DiffUNet(nn.Module):
    def __init__(self, number_modality, number_targets) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(3, number_modality, number_targets, [64, 64, 128, 256, 512, 64])

        self.model = BasicUNetDe(
            3, number_targets, number_targets, [64, 64, 128, 256, 512, 64], 
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False})
        )
    
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
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)
            sample_out = self.sample_diffusion.ddim_sample_loop(
                self.model, 
                (1, number_targets, 96, 96, 96), 
                model_kwargs={"image": image, "embeddings": embeddings}
            )
            sample_out = sample_out["pred_xstart"]
            return sample_out

def parse_args():
    parser = argparse.ArgumentParser(description="BraTS2020 Distributed Training Script")
    
    # Distributed training arguments (handled by torchrun)
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    
    # Custom arguments
    parser.add_argument('--env_type', type=str, default='ddp', help='Environment type for training')
    parser.add_argument('--max_epochs', type=int, default=300, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size per GPU')
    parser.add_argument('--val_every', type=int, default=10, help='Validate every N epochs')
    parser.add_argument('--num_gpus', type=int, default=2, help='Number of GPUs per node')
    parser.add_argument('--logdir', type=str, default="./logs", help='Directory for logs and models')
    parser.add_argument('--master_ip', type=str, default='localhost', help='Master node IP')
    parser.add_argument('--master_port', type=int, default=17751, help='Master node port')
    parser.add_argument('--training_script', type=str, default='train.py', help='Training script name')
    
    args, _ = parser.parse_known_args()
    return args

def main():
    args = parse_args()
    
    # Initialize the trainer
    trainer = BraTSTrainer(args)
    
    # Ensure the model save path exists
    model_save_path = os.path.join(args.logdir, "model")
    os.makedirs(model_save_path, exist_ok=True)
    
    # Load datasets
    data_dir = "/kaggle/input/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    train_ds, val_ds, test_ds = get_loader_brats(data_dir=data_dir, batch_size=args.batch_size, fold=0)
    
    # Start training
    trainer.train(train_dataset=train_ds, val_dataset=val_ds)

if __name__ == "__main__":
    main()
