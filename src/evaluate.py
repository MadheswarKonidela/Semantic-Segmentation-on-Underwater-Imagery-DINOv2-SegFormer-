import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import numpy as np
import json

from src.models.segmentation_model import SegmentationModel
from src.data.suim_dataset import SUIMDataset
from src.utils.metrics import calculate_iou, calculate_pixel_accuracy
from src.utils.visualization import save_segmentation_results
from src.utils.config_parser import load_config
from configs.project_settings import ID_TO_CLASS, ID_TO_COLOR # For detailed metrics/visualization

def evaluate(config):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset (using 'TEST' split)
    test_dataset = SUIMDataset(
        data_dir=config.data_root,
        image_size=config.image_size,
        normalize_mean=config.normalize_mean,
        normalize_std=config.normalize_std,
        split='test',
        transform_type='test'
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.val_batch_size, # Use validation batch size for evaluation
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # Initialize model and load trained weights
    model = SegmentationModel(config).to(device)
    model_path = os.path.join(config.save_dir, config.model_name, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model not found at {model_path}. Please train the model first.")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path} for evaluation.")

    all_preds = []
    all_targets = []
    
    # Setup visualization directory
    vis_dir = os.path.join("results/visualizations", config.model_name + "_predictions")
    os.makedirs(vis_dir, exist_ok=True)

    print(f"Starting evaluation for {config.model_name} on TEST set...")
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc=f"Evaluating {config.model_name}")
        # Change this line to also receive img_name
        for batch_idx, (images, masks, img_names) in enumerate(test_pbar): 
            images_gpu, masks_gpu = images.to(device), masks.to(device)
            outputs = model(images_gpu)
            
            preds = torch.argmax(outputs, dim=1) # Get class predictions
            all_preds.append(preds.cpu())
            all_targets.append(masks_gpu.cpu())

            # Save visualizations for a few samples per batch (or all if desired)
            # You might want to save more systematically, e.g., N samples total
            # For demonstration, saving first few samples from batches
            if batch_idx < 5: # Save visualization for the first 5 batches
                for i in range(images.shape[0]):
                    # Denormalize image for saving
                    img_denormalized = images[i].permute(1, 2, 0).numpy() # C, H, W -> H, W, C
                    img_denormalized = img_denormalized * np.array(config.normalize_std) + np.array(config.normalize_mean)
                    img_denormalized = (img_denormalized * 255).astype(np.uint8)

                    # Use the actual image filename
                    current_img_name = img_names[i] 
                    try:
                        save_segmentation_results(
                            original_image=img_denormalized,
                            ground_truth_mask=masks_gpu[i].cpu().numpy(),
                            prediction_mask=preds[i].cpu().numpy(),
                            output_dir=vis_dir,
                            img_name=current_img_name, # Use actual filename here
                            num_classes=config.num_classes
                        )
                        print(f"Successfully saved {current_img_name}.")
                    except Exception as e:
                        print(f"Failed to save visualization for {current_img_name}. Error: {e}")
                        import traceback
                        traceback.print_exc()
                    

    # Concatenate all predictions and targets for final metric calculation
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate metrics
    iou_per_class, mean_iou = calculate_iou(all_preds, all_targets, config.num_classes)
    pixel_accuracy = calculate_pixel_accuracy(all_preds, all_targets)

    print("\n--- Evaluation Results ---")
    print(f"Overall Pixel Accuracy: {pixel_accuracy:.4f}")
    print(f"Mean IoU (mIoU): {mean_iou:.4f}")
    print("IoU per class:")
    metrics_data = {
        "model_name": config.model_name,
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": mean_iou,
        "iou_per_class": {}
    }
    for i, iou in enumerate(iou_per_class):
        class_name = ID_TO_CLASS.get(i, f"Class_{i}")
        print(f"  {class_name}: {iou:.4f}" if not np.isnan(iou) else f"  {class_name}: N/A (not present)")
        metrics_data["iou_per_class"][class_name] = iou if not np.isnan(iou) else None

    # Save metrics to a JSON file
    metrics_save_path = os.path.join("results/metrics", f"{config.model_name}_metrics.json")
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Metrics saved to: {metrics_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained semantic segmentation model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the model configuration YAML file.")
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate(config)