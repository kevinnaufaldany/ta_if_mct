import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pycocotools import mask as coco_mask
import json
import os
from datetime import datetime


def check_set_gpu(override=None):
    if override == None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"Using MPS: {torch.backends.mps.is_available()}")
        else:
            device = torch.device('cpu')
            print(f"Using CPU: {torch.device('cpu')}")
    else:
        device = torch.device(override)
    return device


def save_checkpoint(model, optimizer, epoch, fold, loss, save_dir='checkpoints', filename=None):
    """
    Save model checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if filename is None:
        filename = f'fold{fold}_epoch{epoch}_loss{loss:.4f}.pth'
    
    filepath = os.path.join(save_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'fold': fold,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")
    
    return filepath
 

def load_checkpoint(model, checkpoint_path, optimizer=None, device='cuda'):
    """
    Returns:
        epoch, fold, loss
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    fold = checkpoint.get('fold', 0)
    loss = checkpoint.get('loss', 0.0)
    
    print(f"Checkpoint loaded: epoch {epoch}, fold {fold}, loss {loss:.4f}")
    
    return epoch, fold, loss


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    iou = intersection / union
    return iou


def calculate_batch_iou_map(predictions, targets, score_threshold=0.5):
    # COCO 2025 Standard: 10 IoU thresholds from 0.50 to 0.95 (step 0.05)
    iou_thresholds = np.arange(0.50, 1.00, 0.05)  # [0.50, 0.55, 0.60, ..., 0.95]
    
    # Storage for AP at each threshold
    ap_per_threshold = {f"{thresh:.2f}": [] for thresh in iou_thresholds}
    
    # Storage for IoU at specific thresholds
    iou_values_per_threshold = {f"{thresh:.2f}": [] for thresh in iou_thresholds}
    
    # Process each image pair
    for pred, target in zip(predictions, targets):
        if 'masks' not in pred or 'masks' not in target:
            continue
        
        pred_masks = pred['masks'].cpu().numpy()
        target_masks = target['masks'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        
        # Filter predictions by confidence threshold
        keep = pred_scores >= score_threshold
        if keep.sum() == 0:
            continue
        
        pred_masks = pred_masks[keep]
        pred_scores = pred_scores[keep]
        
        if len(target_masks) == 0 or len(pred_masks) == 0:
            continue
        
        # Compute IoU matrix: [num_preds x num_targets]
        iou_matrix = np.zeros((len(pred_masks), len(target_masks)))
        for i, pred_mask in enumerate(pred_masks):
            pred_binary = (pred_mask[0] > 0.5)
            for j, target_mask in enumerate(target_masks):
                target_binary = (target_mask > 0)
                iou_matrix[i, j] = calculate_iou(pred_binary, target_binary)
        
        if iou_matrix.size == 0:
            continue
        
        # Sort predictions by confidence (descending) for AP calculation
        sorted_idx = np.argsort(pred_scores)[::-1]
        
        # Best IoU for each prediction (for IoU metrics)
        best_ious = iou_matrix.max(axis=1)
        
        # ===== Calculate AP and IoU for each threshold =====
        for thresh in iou_thresholds:
            thresh_key = f"{thresh:.2f}"
            
            # --- IoU Metric at this threshold ---
            # Only count IoUs from predictions that match >= threshold
            matched_ious = best_ious[best_ious >= thresh]
            iou_values_per_threshold[thresh_key].extend(matched_ious.tolist())
            
            # --- AP Metric at this threshold ---
            matched_targets = np.zeros(len(target_masks), dtype=bool)
            tp_list = []
            fp_list = []
            
            for idx in sorted_idx:
                best_iou_idx = iou_matrix[idx].argmax()
                best_iou = iou_matrix[idx, best_iou_idx]
                
                # True Positive if IoU >= threshold and target not already matched
                if best_iou >= thresh and not matched_targets[best_iou_idx]:
                    matched_targets[best_iou_idx] = True
                    tp_list.append(1)
                    fp_list.append(0)
                else:
                    tp_list.append(0)
                    fp_list.append(1)
            
            # Calculate AP using precision-recall curve
            if len(tp_list) > 0:
                tp = np.cumsum(tp_list)
                fp = np.cumsum(fp_list)
                recall = tp / max(len(target_masks), 1)
                precision = tp / (tp + fp + 1e-10)
                
                # AP = Area under precision-recall curve
                ap = np.trapz(precision, recall)
                ap_per_threshold[thresh_key].append(ap)
    
    # ===== Aggregate metrics across all images =====
    
    # AP@0.50:0.95 (PRIMARY METRIC): Mean of AP across all 10 thresholds
    ap_values = []
    for thresh in iou_thresholds:
        thresh_key = f"{thresh:.2f}"
        if len(ap_per_threshold[thresh_key]) > 0:
            ap_values.append(np.mean(ap_per_threshold[thresh_key]))
        else:
            ap_values.append(0.0)
    
    ap_50_95 = np.mean(ap_values)  # COCO PRIMARY METRIC
    
    # AP@0.50 (for YOLO/Faster R-CNN comparison)
    ap_50 = ap_values[0] if len(ap_values) > 0 else 0.0
    
    # AP@0.75 (strict localization)
    ap_75_idx = np.where(np.abs(iou_thresholds - 0.75) < 0.01)[0]
    ap_75 = ap_values[ap_75_idx[0]] if len(ap_75_idx) > 0 else 0.0
    
    # IoU@0.50:0.95: Mean IoU across all 10 thresholds
    iou_values = []
    for thresh in iou_thresholds:
        thresh_key = f"{thresh:.2f}"
        if len(iou_values_per_threshold[thresh_key]) > 0:
            iou_values.append(np.mean(iou_values_per_threshold[thresh_key]))
        else:
            iou_values.append(0.0)
    
    iou_50_95 = np.mean(iou_values)  # Modern replacement for "Mean IoU"
    
    # IoU@0.50 (additional reference)
    iou_50 = iou_values[0] if len(iou_values) > 0 else 0.0
    
    # IoU@0.75 (additional reference)
    iou_75_idx = np.where(np.abs(iou_thresholds - 0.75) < 0.01)[0]
    iou_75 = iou_values[iou_75_idx[0]] if len(iou_75_idx) > 0 else 0.0
    
    # ===== Return COCO 2025 Standard Metrics =====
    metrics = {
        # PRIMARY METRIC (wajib bold di paper Table 1)
        "AP@0.50:0.95": ap_50_95,
        
        # Secondary AP metrics
        "AP@0.50": ap_50,        # For comparison with YOLOv5/v7/v8, Faster R-CNN
        "AP@0.75": ap_75,        # Strict localization benchmark
        
        # IoU metrics (modern, NO "Mean IoU" terminology)
        "IoU@0.50:0.95": iou_50_95,  # Replacement for old "Mean IoU"
        "IoU@0.50": iou_50,          # Additional reference
        "IoU@0.75": iou_75,          # Additional reference
    }
    
    return metrics


def calculate_map(predictions, targets, iou_threshold=0.5):
    # Simplified mAP calculation
    # Untuk implementasi lengkap, gunakan pycocotools
    
    aps = []
    
    for pred, target in zip(predictions, targets):
        if len(pred['masks']) == 0 or len(target['masks']) == 0:
            continue
        
        pred_masks = pred['masks'].cpu().numpy()
        target_masks = target['masks'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        
        # Sort predictions by score
        sorted_idx = np.argsort(pred_scores)[::-1]
        pred_masks = pred_masks[sorted_idx]
        pred_scores = pred_scores[sorted_idx]
        
        # Match predictions to targets
        matched = np.zeros(len(target_masks), dtype=bool)
        true_positives = []
        false_positives = []
        
        for pred_mask in pred_masks:
            best_iou = 0
            best_idx = -1
            
            for i, target_mask in enumerate(target_masks):
                if matched[i]:
                    continue
                
                iou = calculate_iou(pred_mask > 0.5, target_mask > 0)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            if best_iou >= iou_threshold and best_idx >= 0:
                matched[best_idx] = True
                true_positives.append(1)
                false_positives.append(0)
            else:
                true_positives.append(0)
                false_positives.append(1)
        
        # Calculate precision-recall
        tp = np.cumsum(true_positives)
        fp = np.cumsum(false_positives)
        recall = tp / len(target_masks)
        precision = tp / (tp + fp + 1e-6)
        
        # Calculate AP
        ap = np.trapz(precision, recall)
        aps.append(ap)
    
    if len(aps) == 0:
        return 0.0
    
    return np.mean(aps)

def save_training_history(history, save_path='training_history.json'):
    # Convert numpy arrays and tensors to lists
    history_serializable = {}
    for key, value in history.items():
        if isinstance(value, (np.ndarray, torch.Tensor)):
            history_serializable[key] = value.tolist()
        elif isinstance(value, list):
            history_serializable[key] = [
                v.item() if isinstance(v, (np.generic, torch.Tensor)) else v 
                for v in value
            ]
        else:
            history_serializable[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(history_serializable, f, indent=2)
    
    print(f"Training history saved: {save_path}")

class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0, mode='min', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        if self.mode == 'min':
            self.monitor_op = lambda a, b: a < b - self.min_delta
        elif self.mode == 'max':
            self.monitor_op = lambda a, b: a > b + self.min_delta
        else:
            raise ValueError(f"mode {self.mode} is unknown!")
    
    def __call__(self, score, epoch):
        if self.patience == 0:
            # Early stopping disabled
            return False
        
        if self.best_score is None:
            # First epoch
            self.best_score = score
            self.best_epoch = epoch
            if self.verbose:
                print(f"  [EarlyStopping] Baseline set: {score:.4f}")
        elif self.monitor_op(score, self.best_score):
            # Improvement detected
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"  [EarlyStopping] Improvement detected: {score:.4f} (counter reset)")
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] No improvement ({self.counter}/{self.patience}). Best: {self.best_score:.4f} @ epoch {self.best_epoch}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n{'='*80}")
                    print(f"EARLY STOPPING TRIGGERED!")
                    print(f"No improvement for {self.patience} consecutive epochs.")
                    print(f"Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
                    print(f"{'='*80}\n")
        
        return self.early_stop
    
    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
