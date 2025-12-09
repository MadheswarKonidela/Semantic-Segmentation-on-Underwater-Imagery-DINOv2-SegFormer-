# File: src/data/suim_dataset.py

import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

from src.data.transforms import get_train_transforms, get_val_transforms
# Assuming project_settings.py contains CLASS_MAPPING as you provided
from configs.project_settings import CLASS_MAPPING 

# --- Pre-process the CLASS_MAPPING once for efficient lookup ---
# This creates a dictionary mapping (R, G, B) tuples to their corresponding Class IDs (0-7)
# Ensure CLASS_MAPPING in project_settings.py has the structure:
# { "Class Name": (class_id_int, [R, G, B]), ... }
RGB_TO_CLASS_ID = {tuple(v[1]): v[0] for v in CLASS_MAPPING.values()}

# Define a default background ID, typically 0, and its color
# This is used for any pixels that might not strictly match one of the defined colors
# (e.g., due to anti-aliasing artifacts if they exist in your masks, or if a pixel is truly unmapped)
DEFAULT_BACKGROUND_ID = 0 
DEFAULT_BACKGROUND_RGB = tuple([0, 0, 0]) # From your CLASS_MAPPING for Background waterbody

class SUIMDataset(Dataset):
    def __init__(self, data_dir, image_size, normalize_mean, normalize_std, split='train_val', transform_type='train'):
        self.image_dir = os.path.join(data_dir, split, 'images')
        self.mask_dir = os.path.join(data_dir, split, 'masks') 
        
        # Filter for common image extensions
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

        self.image_size = image_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        if transform_type == 'train':
            self.transform = get_train_transforms(image_size, normalize_mean, normalize_std)
        else: # 'val' or 'test'
            self.transform = get_val_transforms(image_size, normalize_mean, normalize_std)

        print(f"Loaded {len(self.image_files)} images from {self.image_dir}")
        print(f"Expecting masks in {self.mask_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        base_name = os.path.splitext(img_name)[0]
        mask_name = base_name + '.bmp' 
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB (H, W, C)

        # --- IMPORTANT CHANGE: Load mask as color image (BGR by default with cv2) ---
        mask_color = cv2.imread(mask_path, cv2.IMREAD_COLOR) 
        if mask_color is None:
            raise FileNotFoundError(f"Could not load mask: {mask_path}")
        
        mask_color = cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB) # Convert to RGB (H, W, C) for consistency with mapping

        # --- Resize mask to match image dimensions if they are different ---
        # Ensure consistent dimensions before remapping pixel values.
        # Use INTER_NEAREST for masks to preserve discrete class labels during resizing.
        if image.shape[:2] != mask_color.shape[:2]:
            mask_color = cv2.resize(mask_color, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # --- Remap RGB mask pixels to Class IDs (0-7) ---
        # Create an empty single-channel mask (H, W) to store the integer class IDs.
        # Initialize with the default background ID.
        mask_id = np.full(mask_color.shape[:2], fill_value=DEFAULT_BACKGROUND_ID, dtype=np.uint8) 

        # Iterate through the pre-computed RGB_TO_CLASS_ID map and apply the remapping.
        # This correctly handles multiple objects of different colors within a single mask.
        for rgb_color_tuple, class_id_int in RGB_TO_CLASS_ID.items():
            # Find all pixels in the mask_color image that exactly match the current RGB_color_tuple
            # np.all(..., axis=2) checks if all three (R, G, B) channels match for a given pixel.
            # Then, assign the corresponding class_id_int to those pixels in mask_id.
            mask_id[np.all(mask_color == np.array(rgb_color_tuple), axis=2)] = class_id_int
        
        # Optional: Debugging line to check unique values after remapping
        # print(f"DEBUG: Unique class IDs in remapped mask {mask_name}: {np.unique(mask_id)}")


        # Apply transformations (albumentations expects image and mask as numpy arrays)
        # Albumentations will resize both image and mask to config.image_size (e.g., [518, 518])
        augmented = self.transform(image=image, mask=mask_id) # Use mask_id (single channel, 0-7) here
        image = augmented['image']
        mask = augmented['mask'].long() # Masks should be LongTensor for CrossEntropyLoss

        return image, mask, img_name