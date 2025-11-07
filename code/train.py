import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
import numpy as np
import os
import argparse
import json
import random
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import cv2

import wandb
import time
from model import get_model, count_parameters
from datareaders import prepare_kfold_datasets, get_dataloader
from utils import (
    check_set_gpu, save_checkpoint,
    calculate_batch_iou_map, AverageMeter, 
    EarlyStopping
)

def visualize_predictions(model, dataset, device, save_dir, fold_idx, num_samples=5, 
                         confidence_threshold=0.5, seed=42):
    """
    Visualize model predictions vs ground truth untuk validasi. 
    Walau masih berantakan hasil visualisasinya
    """
    model.eval()
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Select random samples
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx in indices:
            img, target = dataset[idx]
            
            # Get prediction
            img_input = img.unsqueeze(0).to(device)
            prediction = model(img_input)[0]
            
            # Convert image to numpy for visualization
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8).copy()
            
            # Create figure with 2 subplots
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            
            # --- Left: Predictions ---
            ax = axes[0]
            ax.imshow(img_np)
            ax.set_title('Predictions - Cassiterite Detection', fontsize=14, fontweight='bold')
            ax.axis('off')
            
            # Filter predictions by confidence
            pred_boxes = prediction['boxes'].cpu().numpy()
            pred_scores = prediction['scores'].cpu().numpy()
            pred_masks = prediction['masks'].cpu().numpy()
            
            valid_idx = pred_scores >= confidence_threshold
            pred_boxes = pred_boxes[valid_idx]
            pred_scores = pred_scores[valid_idx]
            pred_masks = pred_masks[valid_idx]
            
            # Draw predictions
            for box, score, mask in zip(pred_boxes, pred_scores, pred_masks):
                # Draw mask
                mask_binary = (mask[0] > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    contour = contour.squeeze()
                    if contour.ndim == 2:
                        ax.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2, alpha=0.8)
                        ax.fill(contour[:, 0], contour[:, 1], 'r', alpha=0.3)
            
            # Statistics
            total_pred = len(pred_boxes)
            avg_conf = pred_scores.mean() if len(pred_scores) > 0 else 0.0
            fig.text(0.02, 0.5, f'Total: {total_pred}\nAvg Confidence: {avg_conf:.3f}', 
                    ha='left', fontsize=12, fontweight='bold', color='red')
            
            # --- Right: Ground Truth ---
            ax = axes[1]
            ax.imshow(img_np)
            ax.set_title('Ground Truth', fontsize=14, fontweight='bold')
            ax.axis('off')
            
            # Draw ground truth
            gt_boxes = target['boxes'].cpu().numpy()
            gt_masks = target['masks'].cpu().numpy()
            
            for box, mask in zip(gt_boxes, gt_masks):
                # Draw mask
                mask_binary = mask.astype(np.uint8)
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    contour = contour.squeeze()
                    if contour.ndim == 2:
                        ax.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2, alpha=0.8)
                        ax.fill(contour[:, 0], contour[:, 1], 'r', alpha=0.3)
            
            # Calculate metrics for this image (COCO 2025 format)
            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                # Quick IoU calculation - wrap in list for batch function
                metrics = calculate_batch_iou_map([prediction], [target])
                # Use the 0.50:0.95 (COCO primary) metrics for the visualization
                iou_50_95 = metrics['IoU@0.50:0.95']
                ap_50_95 = metrics['AP@0.50:0.95']
                
                # Add metrics text at bottom
                fig.text(0.65, 0.5, 
                        f'Ground Truth: {len(gt_boxes)}\nIoU@0.50:0.95: {iou_50_95:.3f} | AP@0.50:0.95: {ap_50_95:.3f}',
                        ha='right', fontsize=12, fontweight='bold')
            else:
                fig.text(0.65, 0.5, f'Ground Truth: {len(gt_boxes)}', 
                        ha='right', fontsize=12, fontweight='bold', color='red')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'prediction_sample_{idx+1}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"  Saved {len(indices)} prediction visualizations to {save_dir}")


def plot_fold_metrics(history, fold_idx, save_dir):
    """
    Plot metrics untuk satu fold setelah training selesai.
    Buat plot terpisah untuk loss, APm, dan IoU.
    
    Args:
        history: Dict dengan training history
        fold_idx: Fold number (1-based)
        save_dir: Directory untuk save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training and Validation Loss - Fold {fold_idx}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'loss_fold{fold_idx}.png'), dpi=150)
    plt.close()
    
    # 2. Plot AP (COCO 2025 Standard)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, history['AP@0.50'], 'g-', label='AP@0.50', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, history['AP@0.75'], 'b-', label='AP@0.75', linewidth=2, marker='s', markersize=4)
    plt.plot(epochs, history['AP@0.50:0.95'], 'r-', label='AP@0.50:0.95', linewidth=2.5, marker='D', markersize=5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('AP', fontsize=12)
    plt.title(f'AP@0.50:0.95 - Fold {fold_idx}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'ap_fold{fold_idx}.png'), dpi=150)
    plt.close()
    
    # 3. Plot IoU (COCO 2025 Standard)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, history['IoU@0.50'], 'g-', label='IoU@0.50', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, history['IoU@0.75'], 'b-', label='IoU@0.75', linewidth=2, marker='s', markersize=4)
    plt.plot(epochs, history['IoU@0.50:0.95'], 'purple', label='IoU@0.50:0.95', linewidth=2.5, marker='D', markersize=5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('IoU', fontsize=12)
    plt.title(f'IoU@0.50:0.95 - Fold {fold_idx}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'iou_fold{fold_idx}.png'), dpi=150)
    plt.close()
    
    # 4. Plot Primary Metrics Comparison (IoU@0.50:0.95 vs AP@0.50:0.95)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, history['IoU@0.50:0.95'], 'purple', label='IoU@0.50:0.95', linewidth=2.5, marker='D', markersize=6)
    plt.plot(epochs, history['AP@0.50:0.95'], 'red', label='AP@0.50:0.95', linewidth=2.5, marker='o', markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'IoU@0.50:0.95 vs AP@0.50:0.95 - Fold {fold_idx}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'mean_metrics_comparison_fold{fold_idx}.png'), dpi=150)
    plt.close()
    
    print(f"  4 plots saved to {save_dir}")
    print(f"    - loss_fold{fold_idx}.png")
    print(f"    - ap_fold{fold_idx}.png")
    print(f"    - iou_fold{fold_idx}.png")
    print(f"    - mean_metrics_comparison_fold{fold_idx}.png")


def train_one_epoch(model, dataloader, optimizer, device, epoch, scheduler=None,
                   scaler=None, use_amp=False, warmup_iters=0, warmup_factor=0.1, 
                   global_step=0, base_lr=0.0001):
    model.train()
    loss_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, targets) in enumerate(pbar):
        try:
            # Clear cache before processing batch
            if batch_idx > 0:
                torch.cuda.empty_cache()
            
            # Move to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Manual warmup (avoid scheduler warning)
            if global_step < warmup_iters:
                alpha = global_step / warmup_iters
                warmup_lr = warmup_factor * (1 - alpha) + alpha
                for param_group in optimizer.param_groups:
                    param_group['lr'] = base_lr * warmup_lr
            
            # Forward pass with optional AMP
            if use_amp and scaler is not None:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                scaler.scale(losses).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
            
            # Update scheduler per-batch AFTER optimizer.step() (only after warmup)
            if scheduler is not None and global_step >= warmup_iters:
                scheduler.step()
            
            global_step += 1
            
            # Update metrics
            loss_meter.update(losses.item(), len(images))
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
            
            # Delete tensors to free memory
            del images, targets, loss_dict, losses
            if batch_idx % 2 == 0:  # More aggressive cache clearing
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            print(f"\nError at batch {batch_idx}: {str(e)}")
            torch.cuda.empty_cache()
            if "out of memory" in str(e).lower():
                print("CUDA out of memory! Try reducing batch_size to 1 or disable AMP.")
            raise e
    
    return loss_meter.avg, global_step


@torch.no_grad()
def validate(model, dataloader, device, use_amp=False):
    model.eval()
    loss_meter = AverageMeter()
    
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc='Validation')
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Compute loss
        model.train()
        if use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        model.eval()
        
        loss_meter.update(losses.item(), len(images))
        
        # Get predictions
        if use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                predictions = model(images)
        else:
            predictions = model(images)
            
        all_predictions.extend(predictions)
        all_targets.extend(targets)
        
        pbar.set_postfix({'val_loss': f'{loss_meter.avg:.4f}'})
        
        # Clean up memory
        del images, targets, loss_dict, losses, predictions
        if batch_idx % 2 == 0:
            torch.cuda.empty_cache()
    
    # Calculate COCO 2025 Standard Metrics
    # Use low score_threshold (0.05) to capture early training detections
    eval_metrics = calculate_batch_iou_map(all_predictions, all_targets, score_threshold=0.05)
    
    # Return dictionary with COCO-compliant keys
    results = {
        'val_loss': loss_meter.avg,
        
        # PRIMARY METRIC (wajib bold di Table 1)
        'AP@0.50:0.95': eval_metrics['AP@0.50:0.95'],
        
        # Secondary AP metrics
        'AP@0.50': eval_metrics['AP@0.50'],
        'AP@0.75': eval_metrics['AP@0.75'],
        
        # IoU metrics (modern standard)
        'IoU@0.50:0.95': eval_metrics['IoU@0.50:0.95'],
        'IoU@0.50': eval_metrics['IoU@0.50'],
        'IoU@0.75': eval_metrics['IoU@0.75'],
    }
    
    return results


def train_kfold_optimized(config):
    """
    K-Fold Cross-Validation Training - OPTIMIZED VERSION.
    
    Optimizations:
    - Save only best APm model per fold (best_apm_epoch{epoch}.pth)
    - Simplified TensorBoard (optional, lightweight)
    - history_train.json per fold with all metrics
    - Visualizations only at end of fold
    - Clean folder structure: output/fold{N}/
    """
    print("="*80)
    print("K-FOLD CROSS-VALIDATION TRAINING")
    print("="*80)
    print(f"Dataset: {config['data_dir']}")
    print(f"Folds: {config['n_folds']}")
    print(f"Epochs per fold: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Model Type: {config['model_type'].upper()} ")
    print(f"Backbone: {config['backbone'].upper()}")
    print(f"Pretrained: COCO (MaskRCNN_ResNet50_FPN_Weights.COCO_V1)")
    print(f"Trainable layers: {config['trainable_layers']}")
    print(f"Optimizer: {config['optimizer'].upper()}")
    if config['optimizer'] == 'sgd':
        print(f"  Momentum: {config.get('momentum', 0.9)}")
    print(f"Learning Rate: {config['lr']}")
    print(f"Scheduler: CosineAnnealingWarmRestarts (T_0={config.get('COSINE_T0', 20)})")
    print(f"Mixed Precision (AMP): {'Enabled' if config.get('use_amp', False) else 'Disabled'}")
    
    early_stop_patience = config.get('early_stopping_patience', 7)
    if early_stop_patience > 0:
        print(f"Early Stopping: Enabled (patience={early_stop_patience} epochs)")
    else:
        print(f"Early Stopping: Disabled")
    
    print(f"\nOutput Structure:")
    print(f"  Checkpoints: checkpoints/fold{{N}}/best_apm_epoch{{epoch}}.pth")
    print(f"  History JSON: {config['output_dir']}/fold{{N}}/history_train.json")
    print(f"  Visualizations: {config['output_dir']}/fold{{N}}/[loss|ap|iou|mean_metrics_comparison]_fold{{N}}.png")
    print("="*80)
    
    # Device
    device = check_set_gpu()
    
    # Create main directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Prepare K-Fold datasets
    print("\nPreparing K-Fold datasets...")
    fold_datasets = prepare_kfold_datasets(
        root_dir=config['data_dir'],
        n_splits=config['n_folds'],
        target_size=tuple(config['image_size'])
    )
    
    all_fold_results = []
    total_training_start = time.time()
    
    # Train each fold
    for fold_idx, (train_dataset, val_dataset) in enumerate(fold_datasets):
        fold_num = fold_idx + 1
        print("\n" + "="*80)
        print(f"FOLD {fold_num}/{config['n_folds']}")
        print("="*80)

        fold_start_time = time.time()

        wandb.init(
            project="cassiterite-segmentation-ta",
            name=f"Fold-{fold_num}_{config['backbone']}_{config['model_type']}_{config['optimizer']}",
            group=f"{config['backbone']}_{config['optimizer']}",
            job_type=f"{config['model_type']}",
            config=config,
            reinit=True
        )
        
        # Create fold directory (SIMPLIFIED: all in one folder)
        fold_dir = os.path.join(config['output_dir'], f'fold{fold_num}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # DataLoaders
        train_loader = get_dataloader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers']
        )
        
        val_loader = get_dataloader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        )
        
        # Create model
        print(f"Creating {config['model_type']} model...")
        model = get_model(
            num_classes=config['num_classes'],
            model_type=config['model_type'],
            backbone=config['backbone'],
            trainable_layers=config['trainable_layers']
        )
        model = model.to(device)
        count_parameters(model)
        
        # Optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        if config['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                params, 
                lr=config['lr'], 
                momentum=config.get('momentum', 0.9),
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                params, 
                lr=config['lr'], 
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'adam':
            optimizer = optim.Adam(
                params, 
                lr=config['lr'], 
                weight_decay=config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {config['optimizer']}. Use 'sgd', 'adam', or 'adamw'")
        
        # Main scheduler: CosineAnnealingWarmRestarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=config.get('COSINE_T0', 20),
            T_mult=config.get('COSINE_T_MULT', 1),
            eta_min=config.get('COSINE_ETA_MIN', 1e-6)
        )
        
        # Warmup parameters (manual warmup untuk avoid warning)
        warmup_iters = min(1000, len(train_loader))
        warmup_factor = 0.1
        
        # GradScaler
        use_amp = config.get('use_amp', False)
        scaler = torch.amp.GradScaler(device='cuda', enabled=use_amp)
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 7),
            min_delta=0.0001,
            mode='max',
            verbose=True
        )
        
        # Training history
        history = {
            'fold': fold_num,
            'train_loss': [],
            'val_loss': [],
            # COCO 2025 Standard Metrics
            'AP@0.50:0.95': [],  # PRIMARY METRIC
            'AP@0.50': [],
            'AP@0.75': [],
            'IoU@0.50:0.95': [],  # Modern replacement for "Mean IoU"
            'IoU@0.50': [],
            'IoU@0.75': [],
            'lr': []
        }
        
        best_ap = 0.0  # Track best AP@0.50:0.95 (PRIMARY METRIC)
        best_epoch = 0
        global_step = 0  # Track global step for warmup
        
        # Training loop
        print(f"\nStarting training for Fold {fold_num}...")
        
        for epoch in range(1, config['epochs'] + 1):
            print(f"\n--- Epoch {epoch}/{config['epochs']} ---")
            start_time = time.time()

            # Train (scheduler steps per-batch inside train_one_epoch)
            train_loss, global_step = train_one_epoch(
                model, train_loader, optimizer, device, epoch,
                scheduler=scheduler,
                scaler=scaler,
                use_amp=use_amp,
                warmup_iters=warmup_iters,
                warmup_factor=warmup_factor,
                global_step=global_step,
                base_lr=config['lr']
            )
            
            # Validate with COCO 2025 metrics
            val_results = validate(model, val_loader, device, use_amp=use_amp)
            epoch_time = time.time() - start_time
            

            val_loss = val_results['val_loss']
            ap_50_95 = val_results['AP@0.50:0.95']  # PRIMARY METRIC
            
            # Save to history (COCO 2025 format)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['AP@0.50:0.95'].append(ap_50_95)
            history['AP@0.50'].append(val_results['AP@0.50'])
            history['AP@0.75'].append(val_results['AP@0.75'])
            history['IoU@0.50:0.95'].append(val_results['IoU@0.50:0.95'])
            history['IoU@0.50'].append(val_results['IoU@0.50'])
            history['IoU@0.75'].append(val_results['IoU@0.75'])
            history['lr'].append(optimizer.param_groups[0]['lr'])
    
            # === Logging ke Weights & Biases ===
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_results['val_loss'],
                "AP@0.50:0.95": val_results['AP@0.50:0.95'],
                "AP@0.50": val_results['AP@0.50'],
                "AP@0.75": val_results['AP@0.75'],
                "IoU@0.50:0.95": val_results['IoU@0.50:0.95'],
                "IoU@0.50": val_results['IoU@0.50'],
                "IoU@0.75": val_results['IoU@0.75'],
                "LearningRate": optimizer.param_groups[0]['lr'],
                "EpochTime(sec)": epoch_time,
                "Fold": fold_num,
            })
            
            # Print summary (COCO 2025 format)
            print(f"Epoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  AP@0.50: {val_results['AP@0.50']:.4f}")
            print(f"  AP@0.75: {val_results['AP@0.75']:.4f}")
            print(f"  AP@0.50:0.95: {ap_50_95:.4f}")
            print(f"  IoU@0.50:0.95: {val_results['IoU@0.50:0.95']:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Epoch Time: {epoch_time:.2f} seconds")
            
            # Save best model based on PRIMARY METRIC (AP@0.50:0.95)
            if ap_50_95 > best_ap:
                # Delete previous best checkpoint
                if best_epoch > 0:
                    old_checkpoint = os.path.join(
                        config['checkpoint_dir'], 
                        f'fold{fold_num}',
                        f'best_ap_epoch{best_epoch}.pth'
                    )
                    if os.path.exists(old_checkpoint):
                        os.remove(old_checkpoint)
                
                best_ap = ap_50_95
                best_epoch = epoch

                # Create checkpoint directory with backbone, model_type, and optimizer
                model_config_dir = f"{config['backbone']}_{config['model_type']}_{config['optimizer']}"
                fold_checkpoint_dir = os.path.join(config['checkpoint_dir'], model_config_dir, f'fold{fold_num}')
                os.makedirs(fold_checkpoint_dir, exist_ok=True)
                
                save_checkpoint(
                    model, optimizer, epoch, fold_num, val_loss,
                    save_dir=fold_checkpoint_dir,
                    filename=f'best_ap_epoch{epoch}.pth'  # Updated filename
                )
                print(f"  [BEST] AP@0.50:0.95: {best_ap:.4f} at epoch {epoch}")
            
            # Early stopping check (based on PRIMARY METRIC)
            if early_stopping(ap_50_95, epoch):
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Best AP@0.50:0.95: {early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch}\n")
                break
        
        # Save final history
        history['best_AP@0.50:0.95'] = best_ap  # COCO 2025 naming
        history['best_epoch'] = best_epoch
        history['total_epochs'] = len(history['train_loss'])
        
        history_path = os.path.join(fold_dir, 'history_train.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"\nHistory saved: {history_path}")
        
        # Plot metrics at end of fold
        print(f"Generating metric visualizations...")
        plot_fold_metrics(history, fold_num, fold_dir)
        
        # Generate prediction visualizations (5 random samples)
        print(f"Generating prediction visualizations...")
        visualize_predictions(
            model, val_dataset, device, fold_dir, fold_num,
            num_samples=5, confidence_threshold=0.5, seed=config['seed']
        )
        
        all_fold_results.append(history)
        print(f"\nFold {fold_num} completed. Best AP@0.50:0.95: {best_ap:.4f} at epoch {best_epoch}")
        
        fold_time = time.time() - fold_start_time
        print(f" Total Fold {fold_num} Time: {fold_time/60:.2f} minutes")

        wandb.log({"FoldTime(min)": fold_time/60})
        # === Tutup session wandb untuk fold ini ===
        wandb.finish()
    
    # Final summary (COCO 2025 format)
    print("\n" + "="*80)
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print("="*80)

    total_training_time = time.time() - total_training_start
    print(f"\n Total Training Time: {total_training_time/60:.2f} minutes ({total_training_time/3600:.2f} hours)")
    
    best_aps = [fold['best_AP@0.50:0.95'] for fold in all_fold_results]
    best_epochs = [fold['best_epoch'] for fold in all_fold_results]
    
    print(f"\nBest AP@0.50:0.95 per fold:")
    for i, (ap_score, epoch) in enumerate(zip(best_aps, best_epochs)):
        print(f"  Fold {i+1}: {ap_score:.4f} (epoch {epoch})")
    
    print(f"\nOverall Statistics:")
    print(f"  AP@0.50:0.95: {np.mean(best_aps):.4f} +/- {np.std(best_aps):.4f}")
    print(f"  Best fold: Fold {np.argmax(best_aps) + 1} (AP: {np.max(best_aps):.4f})")
    
    # Save summary
    summary = {
        'n_folds': config['n_folds'],
        'best_AP@0.50:0.95': best_aps,  # COCO 2025 naming
        'best_epochs': best_epochs,
        'mean_AP': float(np.mean(best_aps)),
        'std_AP': float(np.std(best_aps)),
        'best_fold': int(np.argmax(best_aps) + 1),
        'config': config
    }
    model_config_dir = f"{config['backbone']}_{config['model_type']}_{config['optimizer']}"
    summary_path = os.path.join(config['output_dir'], model_config_dir, 'kfold_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")
    print("\nK-Fold training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train Amodal Instance Segmentation Model (Optimized)')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='dataset11',
                       help='Path to dataset folder')
    parser.add_argument('--image_size', type=int, nargs=2, default=[1080, 1920],
                       help='Image size (height width)')
    
    # Training mode (K-Fold only)
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of folds for k-fold CV')
    
    # Model
    parser.add_argument('--model_type', type=str, default='amodal', 
                       choices=['standard', 'amodal'],
                       help='Model type: standard or amodal')
    parser.add_argument('--backbone', type=str, default='resnet50_fpn_v2',
                       choices=['resnet50_fpn_v1', 'resnet50_fpn_v2'],
                       help='Backbone architecture: resnet50_fpn_v1 or resnet50_fpn_v2 (all use COCO pretrained)')
    parser.add_argument('--trainable_layers', type=int, default=3,
                       help='Number of trainable layers in backbone (0-5 for ResNet50)')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes (including background)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'adamw'],
                       help='Optimizer type')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum for SGD optimizer')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                       help='Weight decay')
    parser.add_argument('--COSINE_T0', type=int, default=20,
                       help='T_0 for CosineAnnealingWarmRestarts')
    parser.add_argument('--COSINE_T_MULT', type=int, default=1,
                       help='T_mult for CosineAnnealingWarmRestarts')
    parser.add_argument('--COSINE_ETA_MIN', type=float, default=1e-6,
                       help='Minimum learning rate for CosineAnnealingWarmRestarts')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use Automatic Mixed Precision (AMP)')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false',
                       help='Disable AMP')
    parser.add_argument('--early_stopping_patience', type=int, default=7,
                       help='Early stopping patience (0 to disable)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory (contains history/ and visualizations/ subfolders)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Create config dict
    config = vars(args)
    
    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Train K-Fold Cross-Validation
    train_kfold_optimized(config)


if __name__ == '__main__':
    main()
