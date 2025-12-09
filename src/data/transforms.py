
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size, normalize_mean, normalize_std):
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1), # Add a small probability for vertical flips
        A.ShiftScaleRotate( # A powerful combination of shifts, scales, and rotations
            shift_limit=0.0625,  # Shift the image by up to 6.25%
            scale_limit=0.1,     # Scale the image by +/- 10%
            rotate_limit=15,     # Rotate by +/- 15 degrees
            p=0.1
        ),
        # A.RandomSizedCrop(
        #     min_max_height=(int(image_size[0]*0.7), image_size[0]), # Crop to a random size between 70% and 100% of the original height
        #     # height=image_size[0], 
        #     # width=image_size[1],
        #     size=image_size,
        #     p=0.3
        # ),

        # --- PHOTOMETRIC / COLOR AUGMENTATIONS ---
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3), # Increased limits and probability
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3), # Increased limits and probability
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.2), # Additional color jitter
        
        # # --- NOISE / BLUR AUGMENTATIONS ---
        # A.OneOf([
        #     A.GaussNoise(var_limit=(10, 50), p=0.2),
        #     A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        # ], p=0.2),

        A.Normalize(mean=normalize_mean, std=normalize_std),
        ToTensorV2(),
    ])

def get_val_transforms(image_size, normalize_mean, normalize_std):
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=normalize_mean, std=normalize_std),
        ToTensorV2(),
    ])

# Note: Transforms should be applied AFTER JSON to mask conversion in the Dataset class.
# Albumentations handles both image and mask transformations simultaneously if passed correctly.