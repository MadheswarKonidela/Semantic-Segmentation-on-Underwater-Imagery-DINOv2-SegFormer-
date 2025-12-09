import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class DINOv2Backbone(nn.Module):
    def __init__(self, model_name="dinov2_vitl14", freeze=True,block_indices=(4, 6, 8,12),trainable_blocks=(11, 12, 14, 18, 22)):
        super().__init__()
        # Load DINOv2 from torch.hub (or timm if you prefer)
        # timm.create_model supports dinov2 models from FacebookResearch directly.
        # Ensure you have the 'timm' library installed.
        #self.backbone = timm.create_model(model_name, pretrained=True)
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"DINOv2 backbone ({model_name}) parameters frozen.")
        else:
            print(f"DINOv2 backbone ({model_name}) parameters are trainable.")

        # DINOv2 ViT models output patch embeddings.
        # We need to extract features at different scales for segmentation heads.
        # For a ViT-L/14, the output is typically (Batch_Size, Num_Patches, Embedding_Dim).
        # Num_Patches depends on input_resolution / patch_size.
        # For 512x512 input with 14x14 patch size, Num_Patches = (512/14) * (512/14) approx 36*36 = 1296.
        # Embedding_Dim for ViT-L is 1024.

        # We'll expose a method to get feature maps from the last layer.
        # For multi-scale features, you'd need to modify the DINOv2 model or use hooks
        # to capture outputs from intermediate blocks. For simplicity, we'll extract
        # the final layer output and reshape it.
        self.feature_dim = self.backbone.embed_dim # Embedding dimension of the ViT
        self.block_indices = block_indices
        self.trainable_blocks = trainable_blocks
        self.hook_outputs = {}
        
        # Freeze all parameters by default
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze specified blocks
        for idx in trainable_blocks:
            for param in self.backbone.blocks[idx].parameters():
                param.requires_grad = True
            print(f"Block {idx} set to trainable.")
            
        if hasattr(self.backbone, 'norm'):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
            print("Final norm layer set to trainable.")
        
        #Register hooks on intermediate blocks
        for idx in block_indices:
            self.backbone.blocks[idx].register_forward_hook(self._get_hook(idx))

    def _get_hook(self, idx):
        def hook(module, input, output):
            # Output shape: (B, N, C)
            self.hook_outputs[idx] = output
        return hook
    
    def forward(self, x):
        # DINOv2's forward method for Vision Transformers usually returns
        # the class token and patch tokens. We are interested in patch tokens.
        # The output shape is (B, num_patches + 1, embed_dim) where +1 is for CLS token.
        # self.hook_outputs = {}  # Clear stored intermediate features
        
        # Determine the patch grid size based on the input image dimensions
        # This is the correct way to handle non-perfectly-divisible resolutions
        H_img, W_img = x.shape[2:]
        patch_size = self.backbone.patch_size # Should be 14 for ViT-L/14
        H_feat = H_img // patch_size
        W_feat = W_img // patch_size
        final_output = self.backbone.forward_features(x)
        original_h, original_w = x.shape[2:]
        
        patch_tokens = final_output['x_norm_patchtokens']  # (B, N+1, C)
        if hasattr(self.backbone, 'num_prefix_tokens'):
            patch_tokens = patch_tokens[:, self.backbone.num_prefix_tokens:]
        
        sorted_indices = sorted(self.block_indices)
        features = []
        target_resolutions = [
            (original_h // 4, original_w // 4),  # Finer scale
            (original_h // 8, original_w // 8),  # Intermediate scale
            (original_h // 12, original_w // 12), # Coarser scale
            (original_h // 32, original_w // 32)  # Coarsest scale
            ]
        # # Remove the CLS token (if present)
        for i, idx in enumerate(sorted_indices):
            x_block = self.hook_outputs[idx]  # shape: (B, N, C)
            
            if hasattr(self.backbone, 'num_prefix_tokens'):
                x_block = x_block[:, self.backbone.num_prefix_tokens:]
            
            x_block = x_block[:, 1:]
            
            B, N, C = x_block.shape
            x_block = x_block.permute(0, 2, 1).reshape(B, C, H_feat, W_feat)
            # Explicitly resize the feature map to the target resolution
            target_size = target_resolutions[i]
            resized_feat = F.interpolate(x_block, size=target_size, mode='bilinear', align_corners=False)
            features.append(resized_feat)

        # Append final features
        B, N, C = patch_tokens.shape
        H_feat = W_feat = int(N ** 0.5)
        patch_tokens = patch_tokens.permute(0, 2, 1).reshape(B, C, H_feat, W_feat)

        features.append(patch_tokens)  # Add final-layer feature

        return features # Return as a list for compatibility with heads expecting multi-scale

    
    def get_out_channels(self):
        # The number of output channels is the same for all features in this configuration
        # It's based on the backbone's feature_dim
        num_features = len(self.block_indices) + 1 # +1 for the final feature
        out_channels_list = [self.feature_dim] * num_features
        return out_channels_list
# Example of how to use it:
# backbone = DINOv2Backbone()
# dummy_input = torch.randn(1, 3, 512, 512)
# features = backbone(dummy_input)
# print(features[0].shape) # Should be e.g., torch.Size([1, 1024, 36, 36]) for ViT-L/14