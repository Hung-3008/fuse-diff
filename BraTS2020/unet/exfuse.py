import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticSupervision(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SemanticSupervision, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class SemanticEmbeddingBranch(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SemanticEmbeddingBranch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x, high_level_features):
        x = F.relu(self.conv1(x))
        x = x * high_level_features
        x = self.conv2(x)
        return x

class ExplicitChannelResolutionEmbedding(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ExplicitChannelResolutionEmbedding, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.sub_pixel_upsample = nn.PixelShuffle(2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.sub_pixel_upsample(x)
        return x

class DenselyAdjacentPrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DenselyAdjacentPrediction, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, 3, 3, channels // 9, height, width)
        x = x.permute(0, 3, 4, 5, 1, 2).contiguous()
        x = x.view(batch_size, channels // 9, height * 3, width * 3)
        return x

class ExFuse(nn.Module):
    def __init__(self, num_classes):
        super(ExFuse, self).__init__()
        self.encoder = ResNeXt101()
        self.decoder = nn.ModuleList([nn.Conv2d(2048, 512, kernel_size=1),
                                      nn.Conv2d(1024, 256, kernel_size=1),
                                      nn.Conv2d(512, 128, kernel_size=1),
                                      nn.Conv2d(256, 64, kernel_size=1)])
        self.semantic_supervision = SemanticSupervision(256, num_classes)
        self.semantic_embedding_branch = SemanticEmbeddingBranch(64, num_classes)
        self.explicit_channel_resolution_embedding = ExplicitChannelResolutionEmbedding(512, num_classes)
        self.densely_adjacent_prediction = DenselyAdjacentPrediction(64, num_classes)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        x = features[0]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = x + self.decoder[0](features[1])
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = x + self.decoder[1](features[2])
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = x + self.decoder[2](features[3])
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = x + self.decoder[3](features[4])
        x = self.semantic_supervision(x)
        x = self.semantic_embedding_branch(x, features[0])
        x = self.explicit_channel_resolution_embedding(x)
        x = self.densely_adjacent_prediction(x)
        x = self.final_conv(x)
        return x

class ResNeXt101(nn.Module):
    def __init__(self):
        super(ResNeXt101 , self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=1)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=1)
        self.conv5 = nn.Conv2d(1024, 2048, kernel_size=1)
        self.block1 = nn.ModuleList([Bottleneck(64, 256, 1) for _ in range(3)])
        self.block2 = nn.ModuleList([Bottleneck(256, 512, 2) for _ in range(4)])
        self.block3 = nn.ModuleList([Bottleneck(512, 1024, 2) for _ in range(23)])
        self.block4 = nn.ModuleList([Bottleneck(1024, 2048, 2) for _ in range(3)])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        for block in self.block1:
            x = block(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        for block in self.block2:
            x = block(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        for block in self.block3:
            x = block(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        for block in self.block4:
            x = block(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(-1, 2048)
        return [x, self.conv2(x), self.conv3(x), self.conv4(x), self.conv5(x)]

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=stride, padding=1, groups=32)
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        if self.training:
            x = x + residual
        else:
            x = F.relu(x + residual)
        return x