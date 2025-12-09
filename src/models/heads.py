import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetConvBlock(nn.Module):
    """
    Helper block for U-Net decoder: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # IMPORTANT: Verify that the BatchNorm2d takes out_channels as input
        print(f"Init args: in_channels={in_channels}, out_channels={out_channels}")
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class MLPHead(nn.Module):
    """
    A simple MLP head, primarily taking the highest resolution feature map.
    """
    def __init__(self, in_channels, num_classes, mlp_feature_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mlp_feature_dim, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(mlp_feature_dim, num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor], target_size):
        x = features[0] 
        x = self.conv1(x)
        x = self.relu(x)
        output = self.conv2(x)
        output = F.interpolate(output, size=target_size, mode='bilinear', align_corners=False)
        return output

class SegFormerHead(nn.Module):
    """
    SegFormer's lightweight all-MLP decoder for segmentation.
    Expects multi-scale features as input.
    """
    def __init__(self, in_channels_list: list[int], embedding_dim: int, num_classes: int):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.linear_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim, kernel_size=1,bias=False),
                nn.BatchNorm2d(embedding_dim),
                nn.ReLU(inplace=True)
            )
            for in_channels in in_channels_list
        ])

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * len(in_channels_list), embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.dropout = nn.Dropout2d(0.1) 
        self.classifier = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor], target_size):
        H_fuse, W_fuse = features[0].shape[2:]

        projected_upsampled_features = []
        for i, feat in enumerate(features):
            proj_feat = self.linear_projs[i](feat)

            if proj_feat.shape[2:] != (H_fuse, W_fuse):
                proj_feat = F.interpolate(
                    proj_feat,
                    size=(H_fuse, W_fuse),
                    mode='bilinear',
                    align_corners=False
                )
            projected_upsampled_features.append(proj_feat)

        fused_features = torch.cat(projected_upsampled_features, dim=1)
        fused_features = self.linear_fuse(fused_features)
        fused_features = self.dropout(fused_features)
        output = self.classifier(fused_features)
        
        output = F.interpolate(output, size=target_size, mode='bilinear', align_corners=False)
        return output

class UNetHead(nn.Module):
    """
    A U-Net like decoder head.
    This version expects a list of features ordered from finest to coarsest resolution,
    mimicking encoder outputs for skip connections.
    """
    def __init__(self, encoder_channels: list[int], decoder_channels: list[int], num_classes: int):
        super().__init__()
        self.num_decoder_stages = len(decoder_channels)
        
        if len(encoder_channels) != self.num_decoder_stages:
            raise ValueError(f"Number of encoder stages ({len(encoder_channels)}) must match number of decoder stages ({self.num_decoder_stages}).")

        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], decoder_channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True)
        )

        self.up_blocks = nn.ModuleList()
        for i in range(self.num_decoder_stages - 1):
            in_upconv_channels = decoder_channels[i]
            out_upconv_channels = decoder_channels[i+1]
            
            # UNetConvBlock input channels are the sum of upsampled channels and skip channels
            conv_block_in_channels = out_upconv_channels + encoder_channels[self.num_decoder_stages - 2 - i]
            
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_upconv_channels, out_upconv_channels, kernel_size=2, stride=2, bias=False),
                    nn.BatchNorm2d(out_upconv_channels),
                    nn.ReLU(inplace=True),
                    UNetConvBlock(conv_block_in_channels, out_upconv_channels)
                )
            )
        
        self.classifier = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor], target_size):
        reversed_features = features[::-1]

        x = self.bottleneck_conv(reversed_features[0])

        for i in range(self.num_decoder_stages - 1):
            upconv_part = self.up_blocks[i][0:3]
            x = upconv_part(x)

            # Get the skip connection feature map
            skip_feat = reversed_features[i + 1]

            # Match sizes if necessary (e.g., due to odd dimensions)
            if x.shape[2:] != skip_feat.shape[2:]:
                x = F.interpolate(x, size=skip_feat.shape[2:], mode='bilinear', align_corners=False)

            # Concatenate the upsampled tensor with the skip connection
            x = torch.cat([x, skip_feat], dim=1)
            
            assert x.shape[1] == self.up_blocks[i][3].block[0].in_channels, \
                f"Channel mismatch at decoder stage {i}: expected {self.up_blocks[i][3].block[0].in_channels}, got {x.shape[1]}"


            # Apply the convolution block to the concatenated tensor
            conv_block_part = self.up_blocks[i][3]
            x = conv_block_part(x)
            
        output = self.classifier(x)
        
        output = F.interpolate(output, size=target_size, mode='bilinear', align_corners=False)
        return output