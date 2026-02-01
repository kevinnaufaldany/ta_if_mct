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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import get_model
from utils import check_set_gpu


class DetectionError:
    CORRECT = 'correct'
    WRONG_LOCATION = 'wrong_location'
    FALSE_POSITIVE = 'false_positive'
    FALSE_NEGATIVE = 'false_negative'
    LOW_CONFIDENCE = 'low_confidence'


# Color scheme untuk visualisasi
ERROR_COLORS = {
    DetectionError.CORRECT: (0, 255, 0),           # GREEN - Benar
    DetectionError.WRONG_LOCATION: (255, 165, 0),  # ORANGE - Lokasi salah
    DetectionError.FALSE_POSITIVE: (255, 0, 0),    # RED - Over-prediction
    DetectionError.FALSE_NEGATIVE: (0, 0, 255),    # BLUE - Under-prediction
    DetectionError.LOW_CONFIDENCE: (255, 255, 0),  # YELLOW - Confidence rendah
}


def load_model(checkpoint_path, num_classes=2, model_type='standard', backbone='resnet50_fpn_v2', device='cuda'):
    """Load trained model"""
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
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
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


def preprocess_image(image_path):
    """Preprocess image untuk inference - menggunakan ukuran asli"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    transformed = transform(image=image)
    return transformed['image'], image


def load_ground_truth(json_path):
    """Load ground truth - menggunakan ukuran asli tanpa scaling"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    gt_objects = []
    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            points = np.array(shape['points'], dtype=np.float32)
            
            gt_objects.append({
                'points': points.astype(np.int32),
                'label': shape.get('label', 'cassiterite').lower()
            })
    
    return gt_objects


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

# def split_image_4tiles(image, image_tensor):
#     """
#     Split image & tensor menjadi 4 tiles (2x2)
#     Return list of dict:
#     {
#         tile_image,
#         tile_tensor,
#         offset_x,
#         offset_y
#     }
#     """
#     h, w = image.shape[:2]
#     h2, w2 = h // 2, w // 2

#     tiles = []

#     coords = [
#         (0, 0),       # top-left
#         (w2, 0),      # top-right
#         (0, h2),      # bottom-left
#         (w2, h2)      # bottom-right
#     ]

#     for ox, oy in coords:
#         tile_img = image[oy:oy+h2, ox:ox+w2]
#         tile_tensor = image_tensor[:, oy:oy+h2, ox:ox+w2]

#         tiles.append({
#             'image': tile_img,
#             'tensor': tile_tensor,
#             'offset_x': ox,
#             'offset_y': oy
#         })

#     return tiles


def run_inference(model, image_tensor, device, threshold=0.5):
    """Run inference pada image"""
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        predictions = model(image_tensor)[0]
    
    keep = predictions['scores'] > threshold
    
    return {
        'boxes': predictions['boxes'][keep].cpu().numpy(),
        'masks': predictions['masks'][keep].cpu().numpy(),
        'scores': predictions['scores'][keep].cpu().numpy(),
        'labels': predictions['labels'][keep].cpu().numpy()
    }


def Test_matching(predictions, gt_objects, image_shape, 
                          iou_threshold=0.5, confidence_threshold=0.7):
    """
    Test matching antara predictions dan ground truth.
    
    Mengidentifikasi:
    - Correct detections (lokasi benar, confidence tinggi)
    - Wrong location (IoU rendah)
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
            if iou >= iou_threshold:
                if pred_score >= confidence_threshold:
                    error_type = DetectionError.CORRECT
                else:
                    error_type = DetectionError.LOW_CONFIDENCE
            elif iou < iou_threshold:
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


def create_Test_visualization(original_image, predictions, gt_objects, 
                                      matching_result, metrics, save_path):
    """
    Create Test visualization dengan 2 file output:
    1. *_evaluation.png: Predictions | GT (2 panels) dengan keterangan sederhana
    2. *_error_map.png: GT | Error Map (2 panels) dengan legenda
    """
    # ========== FILE 1: Predictions | GT (2 Panels) ==========
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # --- Left Panel: Predictions ---
    ax = axes[0]
    image_pred = original_image.copy()
    
    for pred_mask in predictions['masks']:
        if len(pred_mask.shape) == 3:
            pred_mask = pred_mask[0]
        mask_binary = (pred_mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.squeeze()
            if contour.ndim == 2 and len(contour) >= 3:
                ax.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2, alpha=0.8)
                ax.fill(contour[:, 0], contour[:, 1], 'r', alpha=0.3)
    
    ax.imshow(image_pred)
    ax.set_title('Predictions - Cassiterite Detection', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # --- Right Panel: Ground Truth ---
    ax = axes[1]
    image_gt = original_image.copy()
    
    for gt_obj in gt_objects:
        mask = polygon_to_mask(gt_obj['points'], original_image.shape)
        mask_binary = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.squeeze()
            if contour.ndim == 2:
                ax.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2, alpha=0.8)
                ax.fill(contour[:, 0], contour[:, 1], 'r', alpha=0.3)
    
    ax.imshow(image_gt)
    ax.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Statistics at bottom
    total_pred = metrics['total_pred']
    total_gt = metrics['total_gt']
    avg_conf = metrics['mean_confidence']
    iou_50_95 = metrics.get('IoU@0.50:0.95', metrics['mean_iou'])
    ap_50_95 = metrics.get('AP@0.50:0.95', 0.0)
    
    fig.text(0.02, 0.18, f'Total: {total_pred}\nAvg Confidence: {avg_conf:.3f}', 
            ha='left', va='bottom', fontsize=12, fontweight='bold', color='black')
    
    fig.text(0.98, 0.18, f'Ground Truth: {total_gt}\nIoU@0.50:0.95: {iou_50_95:.3f} | AP@0.50:0.95: {ap_50_95:.3f}', 
            ha='right', va='bottom', fontsize=12, fontweight='bold', color='black')
    
    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")
    
    # ========== FILE 2: GT | Error Map dengan Legenda ==========
    error_map_path = save_path.replace('_evaluation.png', '_error_map.png')
    
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 2, wspace=0.02)  # wspace kecil untuk jarak dekat
    ax_gt = fig.add_subplot(gs[0, 0])
    ax_error = fig.add_subplot(gs[0, 1])
    
    # --- Left Panel: Ground Truth ---
    ax_gt.imshow(original_image)
    ax_gt.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax_gt.axis('off')
    
    for gt_obj in gt_objects:
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
        
        pred_mask = predictions['masks'][pred_idx]
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
    ax_error.set_title('Error Map', fontsize=14, fontweight='bold')
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
    plt.savefig(error_map_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {error_map_path}")


def evaluate_testset(checkpoint_path, testset_dir='testset', output_dir='test_evaluation',
                    threshold=0.5, iou_threshold=0.5, confidence_threshold=0.7,
                    model_type='standard', backbone='resnet50_fpn_v2'):
    """
    Main function untuk Test evaluation
    """
    print("="*90)
    print("Test CASSITERITE DETECTION EVALUATION")
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
    global_metrics = defaultdict(int)
    all_results = []
    
    # Process each image
    for idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(testset_dir, img_file)
        json_path = os.path.join(testset_dir, img_file.replace('.jpg', '.json'))
        
        if not os.path.exists(json_path):
            print(f"⚠ Skipping {img_file} (no JSON)")
            continue
        
        print(f"[{idx}/{len(image_files)}] {img_file}")
        
        # Load & process - menggunakan ukuran asli
        image_tensor, original_image = preprocess_image(img_path)
        gt_objects = load_ground_truth(json_path)
        predictions = run_inference(model, image_tensor, device, threshold)
        
        # Test matching
        matching_result = Test_matching(
            predictions, gt_objects, original_image.shape,
            iou_threshold, confidence_threshold
        )
        
        # Calculate metrics
        metrics = calculate_metrics(matching_result)
        
        # Visualize
        output_path = os.path.join(output_dir, img_file.replace('.jpg', '_evaluation.png'))
        create_Test_visualization(
            original_image, predictions, gt_objects,
            matching_result, metrics, output_path
        )
        
        # Print summary
        # print(f"  Pred: {metrics['total_pred']} | GT: {metrics['total_gt']}")
        # print(f"  ✓ Correct: {metrics['correct']} | ⚠ Low Conf: {metrics['low_confidence']}")
        # print(f"  ❌ FP: {metrics['false_positive']} | FN: {metrics['false_negative']}")
        # print(f"  📍 Wrong Loc: {metrics['wrong_location']}")
        # print(f"  📊 Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | F1: {metrics['f1_score']:.3f}")
        # print(f"  📈 Mean IoU: {metrics['mean_iou']:.3f}\n")
        
        # Accumulate
        for key in ['correct', 'low_confidence', 'wrong_location',
                   'false_positive', 'false_negative', 'total_pred', 'total_gt']:
            global_metrics[key] += metrics[key]
        
        all_results.append(metrics)
    
    # Print global summary
    # print("\n" + "="*90)
    # print("GLOBAL SUMMARY")
    # print("="*90)
    print(f"Total Images: {len(all_results)}")
    print(f"Total Predictions: {global_metrics['total_pred']}")
    print(f"Total Ground Truth: {global_metrics['total_gt']}")
    # print(f"\n📊 Detection Breakdown:")
    # print(f"  ✅ Correct: {global_metrics['correct']}")
    # print(f"  ⚠️  Low Confidence: {global_metrics['low_confidence']}")
    # print(f"  ❌ False Positives: {global_metrics['false_positive']}")
    # print(f"  ❌ False Negatives: {global_metrics['false_negative']}")
    # print(f"  📍 Wrong Location: {global_metrics['wrong_location']}")
    
    tp = global_metrics['correct'] + global_metrics['low_confidence']
    precision = tp / global_metrics['total_pred'] if global_metrics['total_pred'] > 0 else 0
    recall = tp / global_metrics['total_gt'] if global_metrics['total_gt'] > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # print(f"\n📈 Overall Performance:")
    # print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    # print(f"  Recall: {recall:.4f} ({recall*100:.2f}%)")
    # print(f"  F1-Score: {f1:.4f}")
    
    mean_iou = np.mean([r['mean_iou'] for r in all_results])
    print(f"  Mean IoU: {mean_iou:.4f}")
    
    print("="*90)
    print(f"\n✅ Test evaluation completed!")
    print(f"📂 Results saved to: {output_dir}/")
    print("="*90)


def main():
    parser = argparse.ArgumentParser(
        description='Test Evaluation System for Cassiterite Detection'
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
    parser.add_argument('--output_dir', type=str, default='test_evaluation',
                       help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold for predictions')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                       help='IoU threshold for matching')
    parser.add_argument('--confidence_threshold', type=float, default=0.7,
                       help='Threshold untuk high-confidence detection')
    
    args = parser.parse_args()
    
    evaluate_testset(
        checkpoint_path=args.checkpoint,
        testset_dir=args.testset_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        iou_threshold=args.iou_threshold,
        confidence_threshold=args.confidence_threshold,
        model_type=args.model_type,
        backbone=args.backbone
    )


if __name__ == '__main__':
    main()
