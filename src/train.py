import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import argparse
import numpy as np

from src.models.segmentation_model import SegmentationModel
from src.data.suim_dataset import SUIMDataset
from src.utils.metrics import calculate_iou, calculate_pixel_accuracy
from src.utils.config_parser import load_config
from datetime import datetime

def train(config):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up logging and output directories
    #log_dir, checkpoint_dir, vis_dir = setup_logging(config)
    # Load dataset
    full_dataset = SUIMDataset(
        data_dir=config.data_root,
        image_size=config.image_size,
        normalize_mean=config.normalize_mean,
        normalize_std=config.normalize_std,
        split='train_val', # Use the combined split for splitting
        transform_type='train' # Train transforms will be applied here
    )
    # For validation, we might split train_val or use a separate val split if available
    # For simplicity, let's use a small split for validation from train_val for now
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    # Re-instantiate validation dataset with val transforms
    # This is a bit hacky but works for random_split: we assume val_dataset elements are just indices
    # val_dataset_actual = SUIMDataset(
    #     data_dir=config.data_root,
    #     image_size=config.image_size,
    #     normalize_mean=config.normalize_mean,
    #     normalize_std=config.normalize_std,
    #     split='train_val',
    #     transform_type='val'
    # )
    # val_dataset_actual.image_files = [train_dataset.image_files[i] for i in val_dataset.indices]
    
    full_dataset_indices = list(range(len(full_dataset.image_files)))
    train_indices, val_indices = random_split(full_dataset_indices, [train_size, val_size])

    train_dataset = SUIMDataset(
    data_dir=config.data_root,
    image_size=config.image_size,
    normalize_mean=config.normalize_mean,
    normalize_std=config.normalize_std,
    split='train_val', # Split is used to get the base data path
    transform_type='train'
    )
    train_dataset.image_files = [full_dataset.image_files[i] for i in train_indices.indices] # Use the generated indices
    
    val_dataset = SUIMDataset(
    data_dir=config.data_root,
    image_size=config.image_size,
    normalize_mean=config.normalize_mean,
    normalize_std=config.normalize_std,
    split='train_val',
    transform_type='val' # This will use the correct val transforms
    )
    val_dataset.image_files = [full_dataset.image_files[i] for i in val_indices.indices] # Use the generated indices




    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # Initialize model
    model = SegmentationModel(config).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss() # Standard for semantic segmentation
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    # Setup directories for saving
    model_save_path = os.path.join(config.save_dir, config.model_name)
    os.makedirs(model_save_path, exist_ok=True)
    log_file_path = os.path.join("results/training_logs", f"{config.model_name}_log_{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    best_mIoU = -1.0

    print(f"Starting training for {config.model_name}...")
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"Training started for {config.model_name} at {datetime.now()}\n")
        log_file.write(f"Config: {config}\n\n")

        for epoch in range(config.num_epochs):
            model.train()
            total_loss = 0
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]")
            for batch_idx, (images, masks, img_names) in enumerate(train_pbar):
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_train_loss = total_loss / len(train_loader)
            scheduler.step() # Step LR scheduler

            # Validation phase
            model.eval()
            val_total_loss = 0
            all_preds = []
            all_targets = []
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Val]")
                for batch_idx, (images, masks, img_names) in enumerate(val_pbar):
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_total_loss += loss.item()

                    preds = torch.argmax(outputs, dim=1) # Get class predictions
                    all_preds.append(preds.cpu())
                    all_targets.append(masks.cpu())

            avg_val_loss = val_total_loss / len(val_loader)
            
            # Concatenate all predictions and targets for metric calculation
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            # Calculate metrics
            iou_per_class, mean_iou = calculate_iou(all_preds, all_targets, config.num_classes)
            pixel_accuracy = calculate_pixel_accuracy(all_preds, all_targets)

            log_msg = (
                f"Epoch [{epoch+1}/{config.num_epochs}], "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Val mIoU: {mean_iou:.4f}, "
                f"Val Pixel Acc: {pixel_accuracy:.4f}"
            )
            print(log_msg)
            log_file.write(log_msg + '\n')
            log_file.flush()

            # Save best model
            if mean_iou > best_mIoU:
                best_mIoU = mean_iou
                torch.save(model.state_dict(), os.path.join(model_save_path, "best_model.pth"))
                print(f"Saved best model with mIoU: {best_mIoU:.4f}")
                log_file.write(f"Saved best model with mIoU: {best_mIoU:.4f}\n")
                log_file.flush()
            
            # Save checkpoint every few epochs (optional)
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), os.path.join(model_save_path, f"checkpoint_epoch_{epoch+1}.pth"))
                log_file.write(f"Saved checkpoint for epoch {epoch+1}\n")
                log_file.flush()

    print(f"Training finished for {config.model_name}.")
    log_file.write(f"Training finished for {config.model_name} at {datetime.now()}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the model configuration YAML file.")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Ensure save directory exists
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs("results/training_logs", exist_ok=True)

    train(config)