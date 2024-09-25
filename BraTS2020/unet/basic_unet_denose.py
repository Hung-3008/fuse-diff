import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Act, Norm
from monai.utils import ensure_tuple_rep
from typing import Optional, Sequence, Union
from monai.utils import deprecated_arg, ensure_tuple_rep

def nonlinearity(x):
    return F.leaky_relu(x, negative_slope=0.1)

def get_timestep_embedding(timesteps, embedding_dim):
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
        embedding_dim: the dimension of the output.

    Returns:
        An [N x embedding_dim] Tensor of positional embeddings.
    """
    # Compute the sinusoidal embeddings.
    half_dim = embedding_dim // 2
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * 
                    -torch.log(torch.tensor(10000.0)) / (half_dim - 1))
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0,1))
    return emb

class TwoConv(nn.Sequential):
    """two convolutions."""

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        self.temb_proj = torch.nn.Linear(512,
                                         out_chns)

        if dim is not None:
            spatial_dims = dim
        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)
    
    def forward(self, x, temb):
        x = self.conv_0(x)
        x = x + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]
        x = self.conv_1(x)
        return x 

class Down(nn.Module):
    """
    Downsampling with maxpooling then two convolution layers.
    """
    def __init__(self, spatial_dims, in_channels, out_channels, act, norm, bias, dropout):
        super().__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=2)
        self.conv = TwoConv(spatial_dims, in_channels, out_channels, act, norm, bias, dropout)

    def forward(self, x, temb=None):
        x = self.maxpool(x)
        x = self.conv(x, temb)
        return x

class UpCat(nn.Module):
    """
    Upsampling then concatenate with the corresponding encoder output.
    """
    def __init__(self, spatial_dims, in_channels, cat_channels, out_channels, act, norm, bias, dropout, upsample):
        super().__init__()
        self.up = UpSample(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            scale_factor=2,
            mode=upsample,
            bias=bias,
        )
        self.conv = TwoConv(spatial_dims, out_channels + cat_channels, out_channels, act, norm, bias, dropout)

    def forward(self, x, x_e, temb=None):
        x = self.up(x)
        x = torch.cat([x, x_e], dim=1)
        x = self.conv(x, temb)
        return x

class SemanticSupervision(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SemanticSupervision, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class SemanticEmbeddingBranch(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SemanticEmbeddingBranch, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, num_classes, kernel_size=1)

    def forward(self, x, high_level_features):
        x = F.relu(self.conv1(x))
        x = x * high_level_features
        x = self.conv2(x)
        return x

class ExplicitChannelResolutionEmbedding(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ExplicitChannelResolutionEmbedding, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, num_classes * 8, kernel_size=1)
        self.upsample = nn.ConvTranspose3d(num_classes * 8, num_classes, kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.upsample(x)
        return x

class DenselyAdjacentPrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DenselyAdjacentPrediction, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, num_classes, kernel_size=1)
        self.upsample = nn.ConvTranspose3d(num_classes, num_classes, kernel_size=3, stride=3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.upsample(x)
        return x

class BasicUNetDe(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: tuple = (32, 32, 64, 128, 256, 32),
        act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm=("instance", {"affine": True}),
        bias: bool = True,
        dropout: float = 0.0,
        upsample: str = "deconv",
    ):
        super().__init__()

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            nn.Linear(128, 512),
            nn.Linear(512, 512),
        ])

        # Fusion convolutions
        self.conv_fusion0 = TwoConv(spatial_dims, fea[0]*2, fea[0], act, norm, bias, dropout)
        self.conv_fusion1 = TwoConv(spatial_dims, fea[1]*2, fea[1], act, norm, bias, dropout)
        self.conv_fusion2 = TwoConv(spatial_dims, fea[2]*2, fea[2], act, norm, bias, dropout)
        self.conv_fusion3 = TwoConv(spatial_dims, fea[3]*2, fea[3], act, norm, bias, dropout)
        self.conv_fusion4 = TwoConv(spatial_dims, fea[4]*2, fea[4], act, norm, bias, dropout)

        self.conv_0 = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample)

        self.final_conv = Conv["conv", spatial_dims](out_channels, out_channels, kernel_size=1)

        # ExFuse modules
        self.semantic_supervision = SemanticSupervision(fea[2], out_channels)
        self.semantic_embedding_branch = SemanticEmbeddingBranch(fea[5], out_channels)
        self.explicit_channel_resolution_embedding = ExplicitChannelResolutionEmbedding(out_channels, out_channels)
        self.densely_adjacent_prediction = DenselyAdjacentPrediction(out_channels, out_channels)

    def forward(self, x: torch.Tensor, t, embeddings=None):
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        x0 = self.conv_0(x, temb)
        
        if embeddings is not None:
            x0 = torch.cat([x0, embeddings[0]], dim=1)
            x0 = self.conv_fusion0(x0, temb)

        x1 = self.down_1(x0, temb)
        if embeddings is not None:
            x1 = torch.cat([x1, embeddings[1]], dim=1)
            x1 = self.conv_fusion1(x1, temb)

        x2 = self.down_2(x1, temb)
        if embeddings is not None:
            x2 = torch.cat([x2, embeddings[2]], dim=1)
            x2 = self.conv_fusion2(x2, temb)

        # Semantic Supervision
        aux_output = self.semantic_supervision(x2)

        x3 = self.down_3(x2, temb)
        if embeddings is not None:
            x3 = torch.cat([x3, embeddings[3]], dim=1)
            x3 = self.conv_fusion3(x3, temb)

        x4 = self.down_4(x3, temb)
        if embeddings is not None:
            x4 = torch.cat([x4, embeddings[4]], dim=1)
            x4 = self.conv_fusion4(x4, temb)

        u4 = self.upcat_4(x4, x3, temb)
        u3 = self.upcat_3(u4, x2, temb)
        u2 = self.upcat_2(u3, x1, temb)
        u1 = self.upcat_1(u2, x0, temb)

        # Áp dụng Semantic Embedding Branch
        u1 = self.semantic_embedding_branch(u1, x0)

        # Áp dụng Explicit Channel Resolution Embedding
        u1 = self.explicit_channel_resolution_embedding(u1)

        # Áp dụng Densely Adjacent Prediction
        u1 = self.densely_adjacent_prediction(u1)

        # Kết quả cuối cùng
        logits = self.final_conv(u1)

        print(f"Logits shape: {logits.shape}.")
        print(f"Aux_output shape: {aux_output.shape}.") 

        logits = F.interpolate(logits, size=(96, 96, 96), mode='trilinear', align_corners=False)
        aux_output = F.interpolate(aux_output, size=(96, 96, 96), mode='trilinear', align_corners=False)

        print(f"Logits shape after resize: {logits.shape}.")
        print(f"Aux_output shape after resize: {aux_output.shape}.")
        #logits = u1
        

        return logits, aux_output
