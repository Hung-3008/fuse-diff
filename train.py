import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from dataset.brats_data_utils_multi_label import get_loader_brats
from guided_diffusion.gaussian_diffusion import (
    get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
)
from guided_diffusion.resample import UniformSampler
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from monai.inferers import SlidingWindowInferer
from monai.losses.dice import DiceLoss
from monai.utils import set_determinism
from unet.basic_unet import BasicUNetEncoder
from unet.basic_unet_denose import BasicUNetDe
from monai.networks.nets import SegResNet
from monai.networks.layers.factories import Conv
set_determinism(123)


class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(
            3, number_modality, number_targets, [64, 64, 128, 256, 512, 64]
        )

        self.model = BasicUNetDe(
            3, number_modality + number_targets, number_targets,
            [64, 64, 128, 256, 512, 64],
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
            # segresnet_features = self.segresnet(image)
            # combined_features = torch.cat((embeddings, segresnet_features), dim=1)
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)
            sample_out = self.sample_diffusion.ddim_sample_loop(
                self.model, (1, number_targets, 96, 96, 96),
                model_kwargs={"image": image, "embeddings": embeddings}
            )
            sample_out = sample_out["pred_xstart"]
            return sample_out


class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu",
                 val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=17750,
                 training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every,
                         num_gpus, logdir, master_ip, master_port,
                         training_script)
        self.window_infer = SlidingWindowInferer(
            roi_size=[96, 96, 96], sw_batch_size=1, overlap=0.25
        )
        self.model = DiffUNet().to(device)

        self.segresnet = SegResNet(
                        blocks_down=[1, 2, 2, 4],
                        blocks_up=[1, 1, 1],
                        init_filters=16,
                        in_channels=4,  # Input channels set to 4
                        out_channels=64,  # Output channels set to 64
                        dropout_prob=0.2,
                    ).to(device)

        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.segresnet.parameters()), lr=1e-4, weight_decay=1e-3
        )
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.scheduler = LinearWarmupCosineAnnealingLR(
            self.optimizer, warmup_epochs=30, max_epochs=max_epochs
        )

        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)

        
        self.final_conv = Conv["conv", 3](64, 3, kernel_size=1).to(device)
        

    def training_step(self, batch):
        image, label = self.get_input(batch)
        image = image.to(self.device)
        label = label.to(self.device)
        x_start = label

        x_start = (x_start) * 2 - 1
        x_t, t, noise = self.model(x=x_start, pred_type="q_sample")
        pred_tmp, u1 = self.model(x=x_t, step=t, image=image,
                                 pred_type="denoise")
        segnet_features = self.segresnet(image)

        # normalize u1 and segnet_features

        u1_normalized = nn.functional.normalize(u1, p=2, dim=1)
        segnet_features_normalized = nn.functional.normalize(segnet_features, p=2, dim=1)

        pred_xstart = (u1_normalized + segnet_features_normalized) / 2
        pred_xstart = self.final_conv(pred_xstart)
        
        loss_dice = self.dice_loss(pred_xstart, label)
        loss_bce = self.bce(pred_xstart, label)

        pred_xstart = torch.sigmoid(pred_xstart)
        loss_mse = self.mse(pred_xstart, label)

        loss = loss_dice + loss_bce + loss_mse

        self.log("train_loss", loss, step=self.global_step)

        return loss

    def get_input(self, batch):
        image = batch["image"]
        label = batch["label"]

        label = label.float()
        return image, label

    def validation_step(self, batch):
        image, label = self.get_input(batch)
        image = image.to(self.device)
        label = label.to(self.device)

        output = self.window_infer(image, self.model, pred_type="ddim_sample")

        output = torch.sigmoid(output)
        
        output = (output > 0.5).float().cpu().numpy()

        target = label.cpu().numpy()
        # ce
        o = output[:, 1]
        t = target[:, 1]  
        wt = dice(o, t)
        # core
        o = output[:, 0]
        t = target[:, 0]
        tc = dice(o, t)
        # active
        o = output[:, 2]
        t = target[:, 2]
        et = dice(o, t)

        return [wt, tc, et]

    def validation_end(self, mean_val_outputs):
        wt, tc, et = mean_val_outputs

        self.log("wt", wt, step=self.epoch)
        self.log("tc", tc, step=self.epoch)
        self.log("et", et, step=self.epoch)

        self.log("mean_dice", (wt + tc + et) / 3, step=self.epoch)

        mean_dice = (wt + tc + et) / 3
        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(
                self.model,
                os.path.join(model_save_path, f"best_model_{mean_dice:.4f}.pt"),
                delete_symbol="best_model"
            )

        save_new_model_and_delete_last(
            self.model,
            os.path.join(model_save_path, f"final_model_{mean_dice:.4f}.pt"),
            delete_symbol="final_model"
        )

        print(f"wt is {wt}, tc is {tc}, et is {et}, mean_dice is {mean_dice}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on BraTS dataset.")
    parser.add_argument("--data_dir", type=str, default="./data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--max_epoch", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--env", type=str, default="pytorch")

    args = parser.parse_args()

    data_dir = args.data_dir
    logdir = args.logdir
    max_epoch = args.max_epoch
    batch_size = args.batch_size
    val_every = args.val_every
    num_gpus = args.num_gpus
    device = args.device
    env = args.env

    model_save_path = os.path.join(logdir, "model")

    number_modality = 4
    number_targets = 3  # WT, TC, ET

    train_ds, val_ds, test_ds = get_loader_brats(data_dir=data_dir, batch_size=batch_size, fold=0)

    trainer = BraTSTrainer(
        env_type=env,
        max_epochs=max_epoch,
        batch_size=batch_size,
        device=device,
        logdir=logdir,
        val_every=val_every,
        num_gpus=num_gpus,
        master_port=17751,
        training_script=__file__
    )

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)