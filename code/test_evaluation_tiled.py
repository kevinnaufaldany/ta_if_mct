import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import warnings
warnings.filterwarnings('ignore')

import torch
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from collections import defaultdict
from shapely.geometry import Polygon, box

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import get_model
from utils import check_set_gpu


class DetectionError:
    CORRECT = 'correct'
    WRONG_LOCATION = 'wrong_location'
    WRONG_LABEL = 'wrong_label'
    FALSE_POSITIVE = 'false_positive'
    FALSE_NEGATIVE = 'false_negative'
    LOW_CONFIDENCE = 'low_confidence'


# Color scheme untuk visualisasi
ERROR_COLORS = {
    DetectionError.CORRECT: (0, 255, 0),           # GREEN - Benar
    DetectionError.WRONG_LOCATION: (255, 165, 0),  # ORANGE - Lokasi salah
    DetectionError.WRONG_LABEL: (255, 0, 255),     # MAGENTA - Label salah
    DetectionError.FALSE_POSITIVE: (255, 0, 0),    # RED - Over-prediction
    DetectionError.FALSE_NEGATIVE: (0, 0, 255),    # BLUE - Under-prediction
    DetectionError.LOW_CONFIDENCE: (255, 255, 0),  # YELLOW - Confidence rendah
}


def clip_polygon(points, crop_box):
    """
    Clip polygon dengan crop box menggunakan shapely.
    """
    try:
        poly = Polygon(points)
        
        # Fix invalid geometry
        if not poly.is_valid:
            poly = poly.buffer(0)
        
        # Ensure crop_box is valid too
        if not crop_box.is_valid:
            crop_box = crop_box.buffer(0)
        
        clipped = poly.intersection(crop_box)
        
        if clipped.is_empty:
            return None
        if clipped.geom_type == "Polygon":
            return np.array(clipped.exterior.coords)
        elif clipped.geom_type == "MultiPolygon":
            # Ambil polygon terbesar jika hasil intersection adalah MultiPolygon
            largest = max(clipped.geoms, key=lambda p: p.area)
            return np.array(largest.exterior.coords)
        return None
    except Exception as e:
        # Skip polygon yang error
        return None


def load_model(checkpoint_path, num_classes=2, model_type='standard', backbone='resnet50_fpn_v2', device='cuda'):
    """Load trained model"""
    # Normalize path untuk handle Windows backslash
    checkpoint_path = str(Path(checkpoint_path).resolve())
    
    print(f"Loading model from: {checkpoint_path}")
    print(f"Model configuration:")
    print(f"  - Model type: {model_type}")
    print(f"  - Backbone: {backbone}")
    print(f"  - Num classes: {num_classes}")
    
    model = get_model(
        num_classes=num_classes,
        model_type=model_type,
        backbone=backbone,
        trainable_layers=3
    )
    
    # Load checkpoint with explicit file handle to avoid path issues
    with open(checkpoint_path, 'rb') as f:
        checkpoint = torch.load(f, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'N/A')
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        epoch = checkpoint.get('epoch', 'N/A')
    else:
        state_dict = checkpoint
        epoch = 'N/A'
    
    # Remove 'model.' or 'maskrcnn.' prefix if exists
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_key = k.replace('model.', 'maskrcnn.')
        else:
            new_key = k
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded (Epoch: {epoch})\n")
    return model


def split_image_to_tiles(image):
    """
    Split image menjadi 4 tiles (2x2 grid) berdasarkan ukuran asli image
    Returns: list of (tile_image, offset_x, offset_y, tile_id)
    """
    h, w = image.shape[:2]
    tile_h, tile_w = h // 2, w // 2
    
    tiles = []
    tile_id = 1
    
    for i in range(2):  # 2 rows
        for j in range(2):  # 2 cols
            offset_y = i * tile_h
            offset_x = j * tile_w
            
            tile = image[offset_y:offset_y+tile_h, offset_x:offset_x+tile_w]
            tiles.append((tile, offset_x, offset_y, tile_id))
            tile_id += 1
    
    return tiles


def split_ground_truth(gt_objects, tile_bbox):
    """
    Split ground truth berdasarkan tile bounding box
    Args:
        gt_objects: list of dict dengan 'points' dan 'label'
        tile_bbox: (x_min, y_min, x_max, y_max)
    Returns:
        list of clipped gt objects untuk tile ini
    """
    x_min, y_min, x_max, y_max = tile_bbox
    crop_box = box(x_min, y_min, x_max, y_max)
    
    tile_gt = []
    
    for gt_obj in gt_objects:
        clipped_points = clip_polygon(gt_obj['points'], crop_box)
        
        if clipped_points is not None and len(clipped_points) >= 3:
            # Adjust coordinates to tile local coordinates
            local_points = clipped_points.copy()
            local_points[:, 0] -= x_min
            local_points[:, 1] -= y_min
            
            tile_gt.append({
                'points': local_points.astype(np.int32),
                'label': gt_obj['label']
            })
    
    return tile_gt


def preprocess_tile(tile_image):
    """Preprocess tile untuk inference"""
    transform = A.Compose([
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    transformed = transform(image=tile_image)
    return transformed['image']


def load_ground_truth(json_path, target_size=None):
    """Load ground truth, jika target_size diberikan akan di-scale, jika tidak akan menggunakan ukuran asli"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    gt_objects = []
    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            points = np.array(shape['points'], dtype=np.float32)
            
            # Jika target_size diberikan, lakukan scaling
            if target_size is not None:
                orig_h = data.get('imageHeight', target_size[0])
                orig_w = data.get('imageWidth', target_size[1])
                
                scale_h = target_size[0] / orig_h
                scale_w = target_size[1] / orig_w
                
                points[:, 0] *= scale_w
                points[:, 1] *= scale_h
            
            gt_objects.append({
                'points': points.astype(np.int32),
                'label': shape.get('label', 'cassiterite').lower()
            })
    
    return gt_objects


def run_inference_on_tile(model, tile_tensor, device, threshold=0.5):
    """Run inference pada tile"""
    with torch.no_grad():
        tile_tensor = tile_tensor.unsqueeze(0).to(device)
        predictions = model(tile_tensor)[0]
    
    keep = predictions['scores'] > threshold
    
    return {
        'boxes': predictions['boxes'][keep].cpu().numpy(),
        'masks': predictions['masks'][keep].cpu().numpy(),
        'scores': predictions['scores'][keep].cpu().numpy(),
        'labels': predictions['labels'][keep].cpu().numpy()
    }


def merge_tile_predictions(tile_predictions_list, tile_info_list, full_image_size):
    """
    Merge predictions dari semua tiles ke koordinat global
    Args:
        tile_predictions_list: list of predictions dari setiap tile
        tile_info_list: list of (tile_image, offset_x, offset_y, tile_id)
        full_image_size: (height, width) dari full image
    Returns:
        merged predictions dalam koordinat global dengan tile_masks untuk memory efficiency
    """
    merged_boxes = []
    merged_tile_masks = []  # Store tile masks with offsets instead of full masks
    merged_offsets = []  # Store (offset_x, offset_y) for each mask
    merged_scores = []
    merged_labels = []
    
    h, w = full_image_size[:2]
    
    for predictions, (_, offset_x, offset_y, _) in zip(tile_predictions_list, tile_info_list):
        for i in range(len(predictions['boxes'])):
            # Adjust box coordinates
            box = predictions['boxes'][i].copy()
            box[0] += offset_x  # x1
            box[1] += offset_y  # y1
            box[2] += offset_x  # x2
            box[3] += offset_y  # y2
            
            # Store tile mask and offset (memory efficient)
            tile_mask = predictions['masks'][i]
            if len(tile_mask.shape) == 3:
                tile_mask = tile_mask[0]
            
            merged_boxes.append(box)
            merged_tile_masks.append(tile_mask)  # Keep original tile size
            merged_offsets.append((offset_x, offset_y))
            merged_scores.append(predictions['scores'][i])
            merged_labels.append(predictions['labels'][i])
    
    if len(merged_boxes) == 0:
        return {
            'boxes': np.array([]),
            'tile_masks': [],
            'offsets': [],
            'scores': np.array([]),
            'labels': np.array([]),
            'image_size': (h, w)
        }
    
    return {
        'boxes': np.array(merged_boxes),
        'tile_masks': merged_tile_masks,  # List of tile-sized masks
        'offsets': merged_offsets,  # List of (offset_x, offset_y)
        'scores': np.array(merged_scores),
        'labels': np.array(merged_labels),
        'image_size': (h, w)
    }


def calculate_iou(mask1, mask2):
    """Calculate IoU between two binary masks"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    return intersection / union


def polygon_to_mask(points, image_shape):
    """Convert polygon points to binary mask"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask


def calculate_metrics_from_tile_masks(tile_masks, offsets, gt_objects, image_size):
    """
    Calculate metrics dari tile masks tanpa membuat array besar
    Memory efficient version
    """
    h, w = image_size
    
    # Create GT masks
    gt_masks = []
    for gt_obj in gt_objects:
        mask = polygon_to_mask(gt_obj['points'], (h, w))
        gt_masks.append(mask)
    
    if len(tile_masks) == 0 or len(gt_masks) == 0:
        return {'IoU@0.50:0.95': 0.0, 'AP@0.50:0.95': 0.0}
    
    # Calculate IoU incrementally without storing all masks
    all_ious = []
    for tile_mask, (offset_x, offset_y) in zip(tile_masks, offsets):
        # Create full mask for this prediction only
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        tile_h, tile_w = tile_mask.shape
        pred_mask[offset_y:offset_y+tile_h, offset_x:offset_x+tile_w] = (tile_mask > 0.5).astype(np.uint8)
        
        # Calculate IoU with all GT masks
        for gt_mask in gt_masks:
            iou = calculate_iou(pred_mask, gt_mask)
            all_ious.append(iou)
    
    avg_iou = np.mean(all_ious) if all_ious else 0.0
    
    # Simple AP calculation
    thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []
    for thresh in thresholds:
        matches = sum(1 for iou in all_ious if iou >= thresh)
        ap = matches / max(len(tile_masks), len(gt_masks))
        aps.append(ap)
    
    avg_ap = np.mean(aps) if aps else 0.0
    
    return {
        'IoU@0.50:0.95': avg_iou,
        'AP@0.50:0.95': avg_ap
    }


def Test_matching_from_tile_masks(tile_masks, offsets, gt_objects, image_shape, 
                                   pred_scores, pred_labels,
                                   iou_threshold=0.5, confidence_threshold=0.7):
    """
    Test matching untuk merged predictions dengan tile masks (memory efficient)
    """
    h, w = image_shape[:2]
    
    # Convert GT polygons to masks
    gt_masks = []
    for gt_obj in gt_objects:
        mask = polygon_to_mask(gt_obj['points'], image_shape)
        gt_masks.append(mask)
    
    # Convert tile predictions to binary masks (one at a time to save memory)
    pred_masks = []
    for tile_mask, (offset_x, offset_y) in zip(tile_masks, offsets):
        full_mask = np.zeros((h, w), dtype=np.uint8)
        tile_h, tile_w = tile_mask.shape
        full_mask[offset_y:offset_y+tile_h, offset_x:offset_x+tile_w] = (tile_mask > 0.5).astype(np.uint8)
        pred_masks.append(full_mask)
    
    # Label mapping
    label_map = {0: 'background', 1: 'cassiterite'}
    
    # Calculate IoU matrix
    num_pred = len(pred_masks)
    num_gt = len(gt_masks)
    
    iou_matrix = np.zeros((num_pred, num_gt))
    for i in range(num_pred):
        for j in range(num_gt):
            iou_matrix[i, j] = calculate_iou(pred_masks[i], gt_masks[j])
    
    # Matching dengan greedy approach
    matched_predictions = {}
    matched_gt = set()
    
    matches = []
    for i in range(num_pred):
        for j in range(num_gt):
            if iou_matrix[i, j] > 0:
                matches.append((i, j, iou_matrix[i, j]))
    
    matches.sort(key=lambda x: x[2], reverse=True)
    
    for pred_idx, gt_idx, iou_val in matches:
        if pred_idx in matched_predictions or gt_idx in matched_gt:
            continue
        
        matched_predictions[pred_idx] = {
            'gt_idx': gt_idx,
            'iou': iou_val
        }
        matched_gt.add(gt_idx)
    
    # Categorize each prediction
    prediction_errors = []
    
    for i in range(num_pred):
        pred_label = label_map.get(pred_labels[i], 'unknown')
        pred_score = pred_scores[i]
        
        if i in matched_predictions:
            gt_idx = matched_predictions[i]['gt_idx']
            iou = matched_predictions[i]['iou']
            gt_label = gt_objects[gt_idx]['label']
            
            if iou >= iou_threshold and pred_label == gt_label:
                if pred_score >= confidence_threshold:
                    error_type = DetectionError.CORRECT
                else:
                    error_type = DetectionError.LOW_CONFIDENCE
            elif iou >= iou_threshold and pred_label != gt_label:
                error_type = DetectionError.WRONG_LABEL
            elif iou < iou_threshold and pred_label == gt_label:
                error_type = DetectionError.WRONG_LOCATION
            else:
                error_type = DetectionError.FALSE_POSITIVE
        else:
            error_type = DetectionError.FALSE_POSITIVE
        
        prediction_errors.append({
            'pred_idx': i,
            'error_type': error_type,
            'confidence': pred_score,
            'label': pred_label,
            'matched_gt': matched_predictions.get(i, {}).get('gt_idx', None),
            'iou': matched_predictions.get(i, {}).get('iou', 0.0)
        })
    
    # Find false negatives
    gt_errors = []
    for j in range(num_gt):
        if j not in matched_gt:
            gt_errors.append({
                'gt_idx': j,
                'error_type': DetectionError.FALSE_NEGATIVE,
                'label': gt_objects[j]['label']
            })
    
    return {
        'prediction_errors': prediction_errors,
        'gt_errors': gt_errors,
        'num_pred': num_pred,
        'num_gt': num_gt
    }


def Test_matching(predictions, gt_objects, image_shape, 
                          iou_threshold=0.5, confidence_threshold=0.7):
    """
    Test matching antara predictions dan ground truth.
    
    Mengidentifikasi:
    - Correct detections (lokasi benar, label benar, confidence tinggi)
    - Wrong location (IoU rendah tapi label benar)
    - Wrong label (IoU tinggi tapi label salah)
    - False positives (tidak ada GT yang match)
    - False negatives (GT tidak terdeteksi)
    - Low confidence (benar tapi confidence < threshold)
    
    Returns:
        dict dengan detail matching per prediksi dan GT
    """
    h, w = image_shape[:2]
    
    # Convert GT polygons to masks
    gt_masks = []
    for gt_obj in gt_objects:
        mask = polygon_to_mask(gt_obj['points'], image_shape)
        gt_masks.append(mask)
    
    # Convert predictions to masks
    pred_masks = []
    for pred_mask in predictions['masks']:
        if len(pred_mask.shape) == 3:
            pred_mask = pred_mask[0]
        pred_masks.append((pred_mask > 0.5).astype(np.uint8))
    
    # Label mapping (assuming 0=background, 1=cassiterite)
    label_map = {0: 'background', 1: 'cassiterite'}
    
    # Calculate IoU matrix
    num_pred = len(pred_masks)
    num_gt = len(gt_masks)
    
    iou_matrix = np.zeros((num_pred, num_gt))
    for i in range(num_pred):
        for j in range(num_gt):
            iou_matrix[i, j] = calculate_iou(pred_masks[i], gt_masks[j])
    
    # Matching dengan greedy approach
    matched_predictions = {}
    matched_gt = set()
    
    # Sort all possible matches by IoU
    matches = []
    for i in range(num_pred):
        for j in range(num_gt):
            if iou_matrix[i, j] > 0:  # Consider all overlaps
                matches.append((i, j, iou_matrix[i, j]))
    
    matches.sort(key=lambda x: x[2], reverse=True)
    
    # Greedy assignment
    for pred_idx, gt_idx, iou_val in matches:
        if pred_idx in matched_predictions or gt_idx in matched_gt:
            continue
        
        matched_predictions[pred_idx] = {
            'gt_idx': gt_idx,
            'iou': iou_val
        }
        matched_gt.add(gt_idx)
    
    # Categorize each prediction
    prediction_errors = []
    
    for i in range(num_pred):
        pred_label = label_map.get(predictions['labels'][i], 'unknown')
        pred_score = predictions['scores'][i]
        
        if i in matched_predictions:
            gt_idx = matched_predictions[i]['gt_idx']
            iou = matched_predictions[i]['iou']
            gt_label = gt_objects[gt_idx]['label']
            
            # Determine error type
            if iou >= iou_threshold and pred_label == gt_label:
                if pred_score >= confidence_threshold:
                    error_type = DetectionError.CORRECT
                else:
                    error_type = DetectionError.LOW_CONFIDENCE
            elif iou >= iou_threshold and pred_label != gt_label:
                error_type = DetectionError.WRONG_LABEL
            elif iou < iou_threshold and pred_label == gt_label:
                error_type = DetectionError.WRONG_LOCATION
            else:
                error_type = DetectionError.FALSE_POSITIVE
        else:
            error_type = DetectionError.FALSE_POSITIVE
        
        prediction_errors.append({
            'pred_idx': i,
            'error_type': error_type,
            'confidence': pred_score,
            'label': pred_label,
            'matched_gt': matched_predictions.get(i, {}).get('gt_idx', None),
            'iou': matched_predictions.get(i, {}).get('iou', 0.0)
        })
    
    # Find false negatives (unmatched GT)
    gt_errors = []
    for j in range(num_gt):
        if j not in matched_gt:
            gt_errors.append({
                'gt_idx': j,
                'error_type': DetectionError.FALSE_NEGATIVE,
                'label': gt_objects[j]['label']
            })
    
    return {
        'prediction_errors': prediction_errors,
        'gt_errors': gt_errors,
        'num_pred': num_pred,
        'num_gt': num_gt
    }


def calculate_metrics(matching_result):
    """Calculate Test metrics dari matching result"""
    pred_errors = matching_result['prediction_errors']
    gt_errors = matching_result['gt_errors']
    
    # Count by error type
    error_counts = defaultdict(int)
    for pred_err in pred_errors:
        error_counts[pred_err['error_type']] += 1
    for gt_err in gt_errors:
        error_counts[gt_err['error_type']] += 1
    
    # Calculate metrics
    correct = error_counts[DetectionError.CORRECT]
    low_conf = error_counts[DetectionError.LOW_CONFIDENCE]
    wrong_loc = error_counts[DetectionError.WRONG_LOCATION]
    wrong_label = error_counts[DetectionError.WRONG_LABEL]
    fp = error_counts[DetectionError.FALSE_POSITIVE]
    fn = error_counts[DetectionError.FALSE_NEGATIVE]
    
    # True positives = correct + low_confidence (masih benar meski confidence rendah)
    tp = correct + low_conf
    
    # Precision & Recall
    total_pred = matching_result['num_pred']
    total_gt = matching_result['num_gt']
    
    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Mean IoU (only for matched predictions)
    ious = [pred_err['iou'] for pred_err in pred_errors if pred_err['iou'] > 0]
    mean_iou = np.mean(ious) if ious else 0.0
    
    # Mean confidence
    confidences = [pred_err['confidence'] for pred_err in pred_errors]
    mean_confidence = np.mean(confidences) if confidences else 0.0
    
    # Calculate IoU@0.50:0.95 and AP@0.50:0.95 (COCO metrics)
    # IoU at different thresholds (0.50 to 0.95 step 0.05)
    if ious:
        thresholds = np.arange(0.5, 1.0, 0.05)
        iou_at_thresholds = []
        ap_at_thresholds = []
        
        for thresh in thresholds:
            # IoU @ threshold
            iou_at_thresh = np.mean([iou for iou in ious if iou >= thresh])
            iou_at_thresholds.append(iou_at_thresh if iou_at_thresh else 0.0)
            
            # AP @ threshold (recall based)
            matched_at_thresh = sum(1 for iou in ious if iou >= thresh)
            ap = matched_at_thresh / total_gt if total_gt > 0 else 0.0
            ap_at_thresholds.append(ap)
        
        iou_50_95 = np.mean(iou_at_thresholds)
        ap_50_95 = np.mean(ap_at_thresholds)
    else:
        iou_50_95 = 0.0
        ap_50_95 = 0.0
    
    return {
        'correct': correct,
        'low_confidence': low_conf,
        'wrong_location': wrong_loc,
        'wrong_label': wrong_label,
        'false_positive': fp,
        'false_negative': fn,
        'true_positives': tp,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mean_iou': mean_iou,
        'mean_confidence': mean_confidence,
        'total_pred': total_pred,
        'total_gt': total_gt,
        'IoU@0.50:0.95': iou_50_95,
        'AP@0.50:0.95': ap_50_95
    }


def calculate_batch_iou_map(predictions_list, targets_list):
    """
    Calculate IoU and AP metrics untuk batch predictions
    Simple version untuk visualisasi
    """
    if len(predictions_list) == 0:
        return {
            'IoU@0.50:0.95': 0.0,
            'AP@0.50:0.95': 0.0
        }
    
    pred = predictions_list[0]
    target = targets_list[0]
    
    pred_masks = pred['masks']
    gt_masks = target['masks']
    
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return {
            'IoU@0.50:0.95': 0.0,
            'AP@0.50:0.95': 0.0
        }
    
    # Calculate IoU matrix
    ious = []
    for pred_mask in pred_masks:
        if len(pred_mask.shape) == 3:
            pred_mask = pred_mask[0]
        pred_binary = (pred_mask > 0.5).astype(np.uint8)
        
        for gt_mask in gt_masks:
            gt_binary = gt_mask.astype(np.uint8)
            iou = calculate_iou(pred_binary, gt_binary)
            ious.append(iou)
    
    avg_iou = np.mean(ious) if ious else 0.0
    
    # Simple AP calculation (matching at different thresholds)
    thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []
    for thresh in thresholds:
        matches = sum(1 for iou in ious if iou >= thresh)
        ap = matches / max(len(pred_masks), len(gt_masks))
        aps.append(ap)
    
    avg_ap = np.mean(aps) if aps else 0.0
    
    return {
        'IoU@0.50:0.95': avg_iou,
        'AP@0.50:0.95': avg_ap
    }


def visualize_tile_prediction(tile_image, tile_pred, tile_gt, tile_id, matching_result, metrics, save_path):
    """
    Visualize single tile prediction
    Format: Predictions | Ground Truth (2 panels)
    Error Map dibuat terpisah
    """
    # === Main Visualization: 2 Panels ===
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # --- Left Panel: Predictions ---
    ax = axes[0]
    img_np = tile_image.copy()
    ax.imshow(img_np)
    ax.set_title(f'Tile {tile_id} - Predictions - Cassiterite Detection', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Draw predictions
    for pred_mask in tile_pred['masks']:
        if len(pred_mask.shape) == 3:
            pred_mask = pred_mask[0]
        mask_binary = (pred_mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.squeeze()
            if contour.ndim == 2 and len(contour) >= 3:
                ax.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2, alpha=0.8)
                ax.fill(contour[:, 0], contour[:, 1], 'r', alpha=0.3)
    
    # --- Right Panel: Ground Truth ---
    ax = axes[1]
    ax.imshow(img_np)
    ax.set_title(f'Tile {tile_id} - Ground Truth', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Draw ground truth
    for gt_obj in tile_gt:
        # Convert polygon to mask then to contours for consistent rendering
        mask = polygon_to_mask(gt_obj['points'], tile_image.shape)
        mask_binary = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.squeeze()
            if contour.ndim == 2:
                ax.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2, alpha=0.8)
                ax.fill(contour[:, 0], contour[:, 1], 'r', alpha=0.3)
    
    # Statistics at bottom
    total_pred = metrics['total_pred']
    total_gt = metrics['total_gt']
    avg_conf = metrics['mean_confidence']
    iou_50_95 = metrics['IoU@0.50:0.95']
    ap_50_95 = metrics['AP@0.50:0.95']
    
    fig.text(0.02, 0.18, f'Total: {total_pred}\nAvg Confidence: {avg_conf:.3f}', 
            ha='left', va='bottom', fontsize=12, fontweight='bold', color='black')
    
    fig.text(0.98, 0.18, f'Ground Truth: {total_gt}\nIoU@0.50:0.95: {iou_50_95:.3f} | AP@0.50:0.95: {ap_50_95:.3f}', 
            ha='right', va='bottom', fontsize=12, fontweight='bold', color='black')
    
    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # === Separate Error Map ===
    error_map_path = save_path.replace('.png', '_error_map.png')
    visualize_error_map_tile(tile_image, tile_pred, tile_gt, tile_id, matching_result, error_map_path)


def visualize_merged_result(original_image, merged_pred, gt_objects, matching_result, metrics, save_path):
    """
    Visualize merged result dari semua tiles
    Format: Predictions | Ground Truth (2 panels)
    Error Map dibuat terpisah
    """
    # === Main Visualization: 2 Panels ===
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # --- Left Panel: Predictions ---
    ax = axes[0]
    img_np = original_image.copy()
    ax.imshow(img_np)
    ax.set_title('Merged - Predictions - Cassiterite Detection', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Draw predictions from tile masks
    tile_masks = merged_pred['tile_masks']
    offsets = merged_pred['offsets']
    
    for tile_mask, (offset_x, offset_y) in zip(tile_masks, offsets):
        mask_binary = (tile_mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.squeeze()
            if contour.ndim == 2 and len(contour) >= 3:
                contour_global = contour.copy()
                contour_global[:, 0] += offset_x
                contour_global[:, 1] += offset_y
                ax.plot(contour_global[:, 0], contour_global[:, 1], 'r-', linewidth=2, alpha=0.8)
                ax.fill(contour_global[:, 0], contour_global[:, 1], 'r', alpha=0.3)
    
    # --- Right Panel: Ground Truth ---
    ax = axes[1]
    ax.imshow(img_np)
    ax.set_title('Merged - Ground Truth', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Draw ground truth
    for gt_obj in gt_objects:
        # Convert polygon to mask then to contours for consistent rendering
        mask = polygon_to_mask(gt_obj['points'], original_image.shape)
        mask_binary = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.squeeze()
            if contour.ndim == 2:
                ax.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2, alpha=0.8)
                ax.fill(contour[:, 0], contour[:, 1], 'r', alpha=0.3)
    
    # Statistics at bottom
    total_pred = metrics['total_pred']
    total_gt = metrics['total_gt']
    avg_conf = metrics['mean_confidence']
    iou_50_95 = metrics['IoU@0.50:0.95']
    ap_50_95 = metrics['AP@0.50:0.95']
    
    fig.text(0.02, 0.18, f'Total: {total_pred}\nAvg Confidence: {avg_conf:.3f}', 
            ha='left', va='bottom', fontsize=12, fontweight='bold', color='black')
    
    fig.text(0.98, 0.18, f'Ground Truth: {total_gt}\nIoU@0.50:0.95: {iou_50_95:.3f} | AP@0.50:0.95: {ap_50_95:.3f}', 
            ha='right', va='bottom', fontsize=12, fontweight='bold', color='black')
    
    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # === Separate Error Map ===
    error_map_path = save_path.replace('.png', '_error_map.png')
    visualize_error_map_merged(original_image, merged_pred, gt_objects, matching_result, metrics, error_map_path)


def visualize_error_map_tile(tile_image, tile_pred, tile_gt, tile_id, matching_result, save_path):
    """
    Visualize error map untuk tile dengan 2 panel: GT + Error Map dengan legenda
    """
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 2, wspace=0.02)  # wspace kecil untuk jarak dekat
    ax_gt = fig.add_subplot(gs[0, 0])
    ax_error = fig.add_subplot(gs[0, 1])
    
    # --- Left Panel: Ground Truth ---
    ax_gt.imshow(tile_image)
    ax_gt.set_title(f'Tile {tile_id} - Ground Truth', fontsize=14, fontweight='bold')
    ax_gt.axis('off')
    
    # Draw ground truth
    for gt_obj in tile_gt:
        # Convert polygon to mask then to contours for consistent rendering
        mask = polygon_to_mask(gt_obj['points'], tile_image.shape)
        mask_binary = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.squeeze()
            if contour.ndim == 2:
                ax_gt.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2, alpha=0.8)
                ax_gt.fill(contour[:, 0], contour[:, 1], 'r', alpha=0.3)
    
    # --- Right Panel: Error Map ---
    image_error = tile_image.copy()
    
    # Count error types
    error_counts = {
        DetectionError.CORRECT: 0,
        DetectionError.LOW_CONFIDENCE: 0,
        DetectionError.WRONG_LOCATION: 0,
        DetectionError.FALSE_POSITIVE: 0,
        DetectionError.FALSE_NEGATIVE: 0
    }
    
    # Draw predictions dengan warna sesuai error type
    for pred_err in matching_result['prediction_errors']:
        pred_idx = pred_err['pred_idx']
        error_type = pred_err['error_type']
        color = ERROR_COLORS[error_type]
        error_counts[error_type] += 1
        
        pred_mask = tile_pred['masks'][pred_idx]
        if len(pred_mask.shape) == 3:
            pred_mask = pred_mask[0]
        mask_binary = (pred_mask > 0.5).astype(np.uint8)
        
        overlay = image_error.copy()
        overlay[mask_binary == 1] = color
        image_error = cv2.addWeighted(image_error, 0.6, overlay, 0.4, 0)
        
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_error, contours, -1, color, 2)
    
    # Draw false negatives
    for gt_err in matching_result['gt_errors']:
        gt_idx = gt_err['gt_idx']
        gt_obj = tile_gt[gt_idx]
        color = ERROR_COLORS[DetectionError.FALSE_NEGATIVE]
        error_counts[DetectionError.FALSE_NEGATIVE] += 1
        
        mask = polygon_to_mask(gt_obj['points'], tile_image.shape)
        overlay = image_error.copy()
        overlay[mask == 1] = color
        image_error = cv2.addWeighted(image_error, 0.6, overlay, 0.4, 0)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_error, contours, -1, color, 2)
    
    ax_error.imshow(image_error)
    ax_error.set_title(f'Tile {tile_id} - Error Map', fontsize=14, fontweight='bold')
    ax_error.axis('off')
    
    # --- Legend di lower right dengan kotak ---
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label=f'Correct ({error_counts[DetectionError.CORRECT]})',
                  markerfacecolor=np.array(ERROR_COLORS[DetectionError.CORRECT])/255, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label=f'Low Conf ({error_counts[DetectionError.LOW_CONFIDENCE]})',
                  markerfacecolor=np.array(ERROR_COLORS[DetectionError.LOW_CONFIDENCE])/255, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label=f'Wrong Loc ({error_counts[DetectionError.WRONG_LOCATION]})',
                  markerfacecolor=np.array(ERROR_COLORS[DetectionError.WRONG_LOCATION])/255, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label=f'False Pos ({error_counts[DetectionError.FALSE_POSITIVE]})',
                  markerfacecolor=np.array(ERROR_COLORS[DetectionError.FALSE_POSITIVE])/255, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label=f'False Neg ({error_counts[DetectionError.FALSE_NEGATIVE]})',
                  markerfacecolor=np.array(ERROR_COLORS[DetectionError.FALSE_NEGATIVE])/255, markersize=10),
    ]
    
    ax_error.legend(handles=legend_elements, loc='lower right', ncol=5, fontsize=9, frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.12))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_error_map_merged(original_image, merged_pred, gt_objects, matching_result, metrics, save_path):
    """
    Visualize error map untuk merged result dengan 2 panel: GT + Error Map dengan legenda
    """
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 2, wspace=0.02)  # wspace kecil untuk jarak dekat
    ax_gt = fig.add_subplot(gs[0, 0])
    ax_error = fig.add_subplot(gs[0, 1])
    
    # --- Left Panel: Ground Truth ---
    ax_gt.imshow(original_image)
    ax_gt.set_title('Merged - Ground Truth', fontsize=14, fontweight='bold')
    ax_gt.axis('off')
    
    # Draw ground truth
    for gt_obj in gt_objects:
        # Convert polygon to mask then to contours for consistent rendering
        mask = polygon_to_mask(gt_obj['points'], original_image.shape)
        mask_binary = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.squeeze()
            if contour.ndim == 2:
                ax_gt.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2, alpha=0.8)
                ax_gt.fill(contour[:, 0], contour[:, 1], 'r', alpha=0.3)
    
    # --- Right Panel: Error Map ---
    image_error = original_image.copy()
    tile_masks = merged_pred['tile_masks']
    offsets = merged_pred['offsets']
    
    # Count error types
    error_counts = {
        DetectionError.CORRECT: 0,
        DetectionError.LOW_CONFIDENCE: 0,
        DetectionError.WRONG_LOCATION: 0,
        DetectionError.FALSE_POSITIVE: 0,
        DetectionError.FALSE_NEGATIVE: 0
    }
    
    # Draw predictions dengan warna sesuai error type
    for pred_err in matching_result['prediction_errors']:
        pred_idx = pred_err['pred_idx']
        error_type = pred_err['error_type']
        color = ERROR_COLORS[error_type]
        error_counts[error_type] += 1
        
        tile_mask = tile_masks[pred_idx]
        offset_x, offset_y = offsets[pred_idx]
        
        # Create temporary full mask for this prediction only
        h, w = original_image.shape[:2]
        temp_mask = np.zeros((h, w), dtype=np.uint8)
        tile_h, tile_w = tile_mask.shape
        temp_mask[offset_y:offset_y+tile_h, offset_x:offset_x+tile_w] = (tile_mask > 0.5).astype(np.uint8)
        
        overlay = image_error.copy()
        overlay[temp_mask == 1] = color
        image_error = cv2.addWeighted(image_error, 0.6, overlay, 0.4, 0)
        
        contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_error, contours, -1, color, 2)
    
    # Draw false negatives
    for gt_err in matching_result['gt_errors']:
        gt_idx = gt_err['gt_idx']
        gt_obj = gt_objects[gt_idx]
        color = ERROR_COLORS[DetectionError.FALSE_NEGATIVE]
        error_counts[DetectionError.FALSE_NEGATIVE] += 1
        
        mask = polygon_to_mask(gt_obj['points'], original_image.shape)
        overlay = image_error.copy()
        overlay[mask == 1] = color
        image_error = cv2.addWeighted(image_error, 0.6, overlay, 0.4, 0)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_error, contours, -1, color, 2)
    
    ax_error.imshow(image_error)
    ax_error.set_title('Merged - Error Map', fontsize=14, fontweight='bold')
    ax_error.axis('off')
    
    # --- Legend di dalam error map panel ---
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label=f'Correct ({error_counts[DetectionError.CORRECT]})',
                  markerfacecolor=np.array(ERROR_COLORS[DetectionError.CORRECT])/255, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label=f'Low Conf ({error_counts[DetectionError.LOW_CONFIDENCE]})',
                  markerfacecolor=np.array(ERROR_COLORS[DetectionError.LOW_CONFIDENCE])/255, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label=f'Wrong Loc ({error_counts[DetectionError.WRONG_LOCATION]})',
                  markerfacecolor=np.array(ERROR_COLORS[DetectionError.WRONG_LOCATION])/255, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label=f'False Pos ({error_counts[DetectionError.FALSE_POSITIVE]})',
                  markerfacecolor=np.array(ERROR_COLORS[DetectionError.FALSE_POSITIVE])/255, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label=f'False Neg ({error_counts[DetectionError.FALSE_NEGATIVE]})',
                  markerfacecolor=np.array(ERROR_COLORS[DetectionError.FALSE_NEGATIVE])/255, markersize=10),
    ]
    
    ax_error.legend(handles=legend_elements, loc='lower right', ncol=5, fontsize=9, frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.12))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_testset_tiled(checkpoint_path, testset_dir='testset', output_dir='test_evaluation_tiled',
                           threshold=0.5, iou_threshold=0.5, confidence_threshold=0.7,
                           model_type='standard', backbone='resnet50_fpn_v2'):
    """
    Main function untuk tiled test evaluation dengan error analysis
    """
    print("="*90)
    print("TILED CASSITERITE DETECTION EVALUATION")
    print("="*90)
    print(f"Model Type: {model_type}")
    print(f"Backbone: {backbone}")
    print(f"Testset: {testset_dir}")
    print(f"Output: {output_dir}")
    print(f"Confidence threshold: {threshold}")
    print(f"IoU threshold: {iou_threshold}")
    print(f"High confidence threshold: {confidence_threshold}")
    print("="*90 + "\n")
    
    # Setup
    device = check_set_gpu()
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = load_model(checkpoint_path, model_type=model_type, backbone=backbone, device=device)
    
    # Get images
    image_files = sorted([f for f in os.listdir(testset_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"Found {len(image_files)} images\n")
    
    # Global statistics
    global_stats = {
        'total_images': 0,
        'total_pred': 0,
        'total_gt': 0,
        'total_tiles': 0
    }
    
    # Process each image
    for idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(testset_dir, img_file)
        json_path = os.path.join(testset_dir, img_file.replace('.jpg', '.json').replace('.png', '.json'))
        
        if not os.path.exists(json_path):
            print(f"⚠ Skipping {img_file} (no JSON)")
            continue
        
        print(f"[{idx}/{len(image_files)}] {img_file}")
        
        # Create output subdirectory for this image
        img_output_dir = os.path.join(output_dir, img_file.replace('.jpg', '').replace('.png', ''))
        os.makedirs(img_output_dir, exist_ok=True)
        
        # Load image and GT
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Gunakan ukuran asli image tanpa resize
        image_resized = image
        
        gt_objects = load_ground_truth(json_path, target_size=None)
        total_gt_original = len(gt_objects)
        
        # Split image into tiles
        tiles = split_image_to_tiles(image_resized)
        h, w = image_resized.shape[:2]
        tile_h, tile_w = h // 2, w // 2
        
        print(f"  Image size: {h}x{w}, Tile size: {tile_h}x{tile_w}")
        print(f"  Processing {len(tiles)} tiles...")
        
        # Process each tile
        tile_predictions_list = []
        tile_info_list = []
        
        for tile_img, offset_x, offset_y, tile_id in tiles:
            # Get GT for this tile
            tile_bbox = (offset_x, offset_y, 
                        offset_x + tile_w, 
                        offset_y + tile_h)
            tile_gt = split_ground_truth(gt_objects, tile_bbox)
            
            # Preprocess and run inference
            tile_tensor = preprocess_tile(tile_img)
            tile_pred = run_inference_on_tile(model, tile_tensor, device, threshold)
            
            # Test matching for tile
            tile_matching = Test_matching(
                tile_pred, tile_gt, tile_img.shape,
                iou_threshold, confidence_threshold
            )
            
            # Calculate metrics for tile
            tile_metrics = calculate_metrics(tile_matching)
            
            # Save tile visualization
            tile_save_path = os.path.join(img_output_dir, f'tile_{tile_id}.png')
            visualize_tile_prediction(tile_img, tile_pred, tile_gt, tile_id, 
                                     tile_matching, tile_metrics, tile_save_path)
            
            print(f"    Tile {tile_id}: {len(tile_pred['boxes'])} predictions, {len(tile_gt)} GT")
            
            # Store for merging
            tile_predictions_list.append(tile_pred)
            tile_info_list.append((tile_img, offset_x, offset_y, tile_id))
        
        # Merge predictions from all tiles
        print(f"  Merging predictions from all tiles...")
        merged_pred = merge_tile_predictions(tile_predictions_list, tile_info_list, image_resized.shape)
        
        # Test matching for merged result (memory efficient)
        merged_matching = Test_matching_from_tile_masks(
            merged_pred['tile_masks'],
            merged_pred['offsets'],
            gt_objects,
            image_resized.shape,
            merged_pred['scores'],
            merged_pred['labels'],
            iou_threshold,
            confidence_threshold
        )
        
        # Calculate metrics for merged result
        merged_metrics = calculate_metrics(merged_matching)
        
        # Visualize merged result
        merged_save_path = os.path.join(img_output_dir, 'merged_result.png')
        visualize_merged_result(image_resized, merged_pred, gt_objects,
                               merged_matching, merged_metrics, merged_save_path)
        
        # Update global statistics
        total_pred = len(merged_pred['boxes'])
        global_stats['total_images'] += 1
        global_stats['total_pred'] += total_pred
        global_stats['total_gt'] += total_gt_original
        global_stats['total_tiles'] += len(tiles)
        
        print(f"  ✓ Merged result: {total_pred} predictions, {total_gt_original} GT")
        print(f"  📂 Saved to: {img_output_dir}/")
        print()
    
    # Print global summary
    print("\n" + "="*90)
    print("GLOBAL SUMMARY")
    print("="*90)
    print(f"Total Images: {global_stats['total_images']}")
    print(f"Total Tiles Processed: {global_stats['total_tiles']}")
    print(f"Total Predictions (Merged): {global_stats['total_pred']}")
    print(f"Total Ground Truth: {global_stats['total_gt']}")
    print("="*90)
    print(f"\n✅ Tiled evaluation completed!")
    print(f"📂 Results saved to: {output_dir}/")
    print("="*90)


def main():
    parser = argparse.ArgumentParser(
        description='Tiled Test Evaluation System for Cassiterite Detection'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='standard',
                       choices=['standard', 'amodal'],
                       help='Model type: standard or amodal')
    parser.add_argument('--backbone', type=str, default='resnet50_fpn_v2',
                       choices=['resnet50_fpn_v1', 'resnet50_fpn_v2'],
                       help='Backbone architecture')
    parser.add_argument('--testset_dir', type=str, default='testset',
                       help='Directory containing test images and JSONs')
    parser.add_argument('--output_dir', type=str, default='test_evaluation_tiled',
                       help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold for predictions')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                       help='IoU threshold for matching')
    parser.add_argument('--confidence_threshold', type=float, default=0.7,
                       help='Threshold untuk high-confidence detection')
    
    args = parser.parse_args()
    
    # Normalize paths untuk handle Windows backslash
    checkpoint_path = os.path.normpath(args.checkpoint)
    testset_dir = os.path.normpath(args.testset_dir)
    output_dir = os.path.normpath(args.output_dir)
    
    evaluate_testset_tiled(
        checkpoint_path=checkpoint_path,
        testset_dir=testset_dir,
        output_dir=output_dir,
        threshold=args.threshold,
        iou_threshold=args.iou_threshold,
        confidence_threshold=args.confidence_threshold,
        model_type=args.model_type,
        backbone=args.backbone
    )


if __name__ == '__main__':
    main()
