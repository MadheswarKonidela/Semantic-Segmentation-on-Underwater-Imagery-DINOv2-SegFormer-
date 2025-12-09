import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from configs.project_settings import ID_TO_COLOR, UNKNOWN_COLOR

def mask_to_rgb(mask, num_classes):
    """
    Converts a single-channel class-ID mask to an RGB image using defined class colors.
    Args:
        mask (np.ndarray): Single-channel mask with class IDs (H, W).
        num_classes (int): Total number of classes.
    Returns:
        np.ndarray: RGB image (H, W, 3).
    """
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id in range(num_classes):
        color = ID_TO_COLOR.get(class_id, UNKNOWN_COLOR)
        rgb_mask[mask == class_id] = color
    
    # Handle any pixels not mapped (shouldn't happen if all IDs are covered)
    rgb_mask[mask >= num_classes] = UNKNOWN_COLOR

    return rgb_mask

def save_segmentation_results(original_image, ground_truth_mask, prediction_mask, output_dir, img_name, num_classes):
    """
    Saves the original image, ground truth, and prediction side-by-side.
    Args:
        original_image (np.ndarray): Original image (H, W, 3) RGB.
        ground_truth_mask (np.ndarray): Ground truth mask (H, W) class IDs.
        prediction_mask (np.ndarray): Predicted mask (H, W) class IDs.
        output_dir (str): Directory to save the results.
        img_name (str): Original image file name (e.g., 'img_0001.jpg').
        num_classes (int): Total number of classes.
    """
    os.makedirs(output_dir, exist_ok=True)

    gt_rgb = mask_to_rgb(ground_truth_mask, num_classes)
    pred_rgb = mask_to_rgb(prediction_mask, num_classes)

    # Convert original image to uint8 if it's float from normalization (e.g., if you denormalize)
    if original_image.dtype == np.float32:
        original_image = (original_image * 255).astype(np.uint8)

    # Create a composite image
    composite_image = np.hstack((original_image, gt_rgb, pred_rgb))

    plt.figure(figsize=(18, 6)) # Adjust size as needed
    plt.imshow(composite_image)
    plt.title("Original Image | Ground Truth | Prediction")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_segmentation.png"), bbox_inches='tight', dpi=150)
    plt.close()

    # Optionally save individual images if preferred
    # Image.fromarray(original_image).save(os.path.join(output_dir, f"original_{img_name}"))
    # Image.fromarray(gt_rgb).save(os.path.join(output_dir, f"gt_{os.path.splitext(img_name)[0]}.png"))
    # Image.fromarray(pred_rgb).save(os.path.join(output_dir, f"pred_{os.path.splitext(img_name)[0]}.png"))