import torch
import torch.nn as nn
import torch.nn.functional as F # Keep F for potential use in heads, though not used directly here anymore

from src.models.backbone import DINOv2Backbone
from src.models.heads import MLPHead, SegFormerHead, UNetHead

class SegmentationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.num_classes
        self.image_size = config.image_size # (H, W) tuple expected, passed to head for final upsampling

        # Initialize backbone. It's assumed to return a list with a single (B, C, H, W) feature map.
        self.backbone = DINOv2Backbone(
            model_name="dinov2_vitl14", # Use config for model_name
            freeze=True,
            block_indices = (4,6,8,12),
            trainable_blocks=(11, 12, 14, 18, 22)
        )
         
        # Determine input channels for the head from the backbone's output
        # For DINOv2_vitl14, this is 1024.
        # backbone_out_channels = self.backbone.get_out_channels()
        # backbone_feature_dim = self.backbone.feature_dim 
        backbone_feature_dim = self.backbone.get_out_channels()

        # Initialize the appropriate head
        head_type = config.head_type 
        head_params = config.head_params 

        # All heads will receive a list containing one feature map with `backbone_feature_dim` channels.
        # The `in_channels_list` / `encoder_channels` will therefore always be `[backbone_feature_dim]`.
        single_feature_channels_list = backbone_feature_dim

        if head_type == "MLP": 
            # MLPHead is well-suited for a single feature map.
            self.head = MLPHead(in_channels=backbone_feature_dim[-1], # Expects a single int
                                num_classes=self.num_classes, 
                                mlp_feature_dim=head_params.mlp_feature_dim) 

        elif head_type == "SegFormer": 
            # SegFormerHead is typically designed for multi-scale inputs.
            # Passing it a list with only one element `[backbone_feature_dim]` means
            # it will effectively operate on a single scale. Its internal linear projections
            # and fusion might simplify significantly.
            print(f"Warning: SegFormerHead typically expects multi-scale inputs{backbone_feature_dim}")
            self.head = SegFormerHead(in_channels_list=single_feature_channels_list, 
                                       embedding_dim=head_params.embedding_dim, 
                                       num_classes=self.num_classes) 

        elif head_type == "Unet": 
            # UNetHead is critically dependent on multi-scale encoder features for skip connections.
            # Passing it a list with only one element `[backbone_feature_dim]` means
            # its decoder will not have multiple skip connections and will essentially
            # operate as a series of upsampling layers from a single deep feature.
            print(f"Warning: UNetHead typically expects multi-scale encoder features for skip connections. "
                  f"Using with single feature map of {backbone_feature_dim} channels. "
                  f"This may require adjustments to the UNetHead's internal logic for optimal performance.")
            self.head = UNetHead(encoder_channels=single_feature_channels_list, # Assuming it processes a list
                                 decoder_channels=head_params.decoder_channels, 
                                 num_classes=self.num_classes) 
        else: 
            raise ValueError(f"Unknown head type: {head_type}") 

    def forward(self, x): 
        # Extract features from backbone.
        # As per our assumption, `features` will be a list containing a single (B, C, H, W) tensor.
        features = self.backbone(x) 

        # Pass the list of (single) features to the head, along with the original image size for upsampling.
        # The heads are responsible for any internal channel projection or spatial adaptation.
        output = self.head(features, target_size=self.image_size) 
        return output