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


def load_model(checkpoint_path, num_classes=2, device='cuda'):
    """Load trained model"""
    print(f"Loading model from: {checkpoint_path}")
    
    model = get_model(
        num_classes=num_classes,
        model_type='amodal',
        backbone='resnet50',
        pretrained=False
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


def preprocess_image(image_path, target_size=(1080, 1920)):
    """Preprocess image untuk inference"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (target_size[1], target_size[0]))
    
    transform = A.Compose([
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    transformed = transform(image=image_resized)
    return transformed['image'], image_resized


def load_ground_truth(json_path, target_size=(1080, 1920)):
    """Load ground truth dengan scaling ke target size"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    orig_h = data.get('imageHeight', target_size[0])
    orig_w = data.get('imageWidth', target_size[1])
    
    scale_h = target_size[0] / orig_h
    scale_w = target_size[1] / orig_w
    
    gt_objects = []
    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            points = np.array(shape['points'], dtype=np.float32)
            points[:, 0] *= scale_w
            points[:, 1] *= scale_h
            
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
        'total_gt': total_gt
    }


def create_Test_visualization(original_image, predictions, gt_objects, 
                                      matching_result, metrics, save_path):
    """
    Create Test visualization dengan color coding untuk setiap error type.
    Layout: GT | Prediction | Error Map
    """
    fig = plt.figure(figsize=(30, 10.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[15, 1.5], hspace=0.02, wspace=0.1,
                         top=0.98, bottom=0.01, left=0.01, right=0.99)
    
    # === PANEL 1: Ground Truth ===
    ax_gt = fig.add_subplot(gs[0, 0])
    image_gt = original_image.copy()
    
    for gt_obj in gt_objects:
        mask = polygon_to_mask(gt_obj['points'], original_image.shape)
        overlay = image_gt.copy()
        overlay[mask == 1] = (0, 255, 0)  # Green for GT
        image_gt = cv2.addWeighted(image_gt, 0.6, overlay, 0.4, 0)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_gt, contours, -1, (0, 255, 0), 2)
    
    ax_gt.imshow(image_gt)
    ax_gt.set_title(f'Ground Truth ({metrics["total_gt"]} objects)', 
                   fontsize=17, fontweight='bold', pad=12)
    ax_gt.axis('off')
    
    # === PANEL 2: Predictions (All) ===
    ax_pred = fig.add_subplot(gs[0, 1])
    image_pred = original_image.copy()
    
    for pred_mask in predictions['masks']:
        if len(pred_mask.shape) == 3:
            pred_mask = pred_mask[0]
        mask_binary = (pred_mask > 0.5).astype(np.uint8)
        
        overlay = image_pred.copy()
        overlay[mask_binary == 1] = (255, 0, 0)  # Red for predictions
        image_pred = cv2.addWeighted(image_pred, 0.6, overlay, 0.4, 0)
        
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_pred, contours, -1, (255, 0, 0), 2)
    
    ax_pred.imshow(image_pred)
    ax_pred.set_title(f'All Predictions ({metrics["total_pred"]} objects)', 
                     fontsize=17, fontweight='bold', pad=12)
    ax_pred.axis('off')
    
    # === PANEL 3: Error Map (Color Coded) ===
    ax_error = fig.add_subplot(gs[0, 2])
    image_error = original_image.copy()
    
    # Draw predictions dengan warna sesuai error type
    for pred_err in matching_result['prediction_errors']:
        pred_idx = pred_err['pred_idx']
        error_type = pred_err['error_type']
        color = ERROR_COLORS[error_type]
        
        pred_mask = predictions['masks'][pred_idx]
        if len(pred_mask.shape) == 3:
            pred_mask = pred_mask[0]
        mask_binary = (pred_mask > 0.5).astype(np.uint8)
        
        overlay = image_error.copy()
        overlay[mask_binary == 1] = color
        image_error = cv2.addWeighted(image_error, 0.6, overlay, 0.4, 0)
        
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_error, contours, -1, color, 2)
    
    # Draw false negatives (undetected GT)
    for gt_err in matching_result['gt_errors']:
        gt_idx = gt_err['gt_idx']
        gt_obj = gt_objects[gt_idx]
        color = ERROR_COLORS[DetectionError.FALSE_NEGATIVE]
        
        mask = polygon_to_mask(gt_obj['points'], original_image.shape)
        overlay = image_error.copy()
        overlay[mask == 1] = color
        image_error = cv2.addWeighted(image_error, 0.6, overlay, 0.4, 0)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_error, contours, -1, color, 2)
    
    ax_error.imshow(image_error)
    ax_error.set_title('Error Analysis Map (Color Coded)', 
                      fontsize=17, fontweight='bold', pad=12)
    ax_error.axis('off')
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label=f'Correct ({metrics["correct"]})',
                  markerfacecolor=np.array(ERROR_COLORS[DetectionError.CORRECT])/255, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label=f'Low Conf ({metrics["low_confidence"]})',
                  markerfacecolor=np.array(ERROR_COLORS[DetectionError.LOW_CONFIDENCE])/255, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label=f'Wrong Loc ({metrics["wrong_location"]})',
                  markerfacecolor=np.array(ERROR_COLORS[DetectionError.WRONG_LOCATION])/255, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label=f'Wrong Label ({metrics["wrong_label"]})',
                  markerfacecolor=np.array(ERROR_COLORS[DetectionError.WRONG_LABEL])/255, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label=f'False Pos ({metrics["false_positive"]})',
                  markerfacecolor=np.array(ERROR_COLORS[DetectionError.FALSE_POSITIVE])/255, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label=f'False Neg ({metrics["false_negative"]})',
                  markerfacecolor=np.array(ERROR_COLORS[DetectionError.FALSE_NEGATIVE])/255, markersize=10),
    ]
    ax_error.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # === INFO PANEL ===
    ax_info = fig.add_subplot(gs[1, :])
    ax_info.axis('off')
    
    # 3 columns x 3 rows
    col1_x = 0.05
    col2_x = 0.38
    col3_x = 0.71
    
    row_height = 0.30
    row1_y = 0.75
    row2_y = row1_y - row_height
    row3_y = row2_y - row_height
    
    # === Row 1 ===
    ax_info.text(col1_x, row1_y, f"Total Predictions : {metrics['total_pred']}",
                transform=ax_info.transAxes, va='center', ha='left',
                fontfamily='monospace', fontsize=13, color='#CC0000', fontweight='bold')
    
    ax_info.text(col2_x, row1_y, f"Total GT : {metrics['total_gt']}",
                transform=ax_info.transAxes, va='center', ha='left',
                fontfamily='monospace', fontsize=13, color='#00008B', fontweight='bold')
    
    ax_info.text(col3_x, row1_y, f"Precision : {metrics['precision']:.4f}",
                transform=ax_info.transAxes, va='center', ha='left',
                fontfamily='monospace', fontsize=13, color='#006400', fontweight='normal')
    
    # === Row 2 ===
    ax_info.text(col1_x, row2_y, f"Correct : {metrics['correct']}",
                transform=ax_info.transAxes, va='center', ha='left',
                fontfamily='monospace', fontsize=13, color='#228B22', fontweight='normal')
    
    ax_info.text(col2_x, row2_y, f"Mean IoU : {metrics['mean_iou']:.4f}",
                transform=ax_info.transAxes, va='center', ha='left',
                fontfamily='monospace', fontsize=13, color='#800080', fontweight='normal')
    
    ax_info.text(col3_x, row2_y, f"Recall : {metrics['recall']:.4f}",
                transform=ax_info.transAxes, va='center', ha='left',
                fontfamily='monospace', fontsize=13, color='#006400', fontweight='normal')
    
    # === Row 3 ===
    error_summary = f"Errors: FP={metrics['false_positive']} FN={metrics['false_negative']} WL={metrics['wrong_location']} WLabel={metrics['wrong_label']}"
    ax_info.text(col1_x, row3_y, error_summary,
                transform=ax_info.transAxes, va='center', ha='left',
                fontfamily='monospace', fontsize=12, color='#DC143C', fontweight='normal')
    
    ax_info.text(col2_x, row3_y, f"Avg Confidence : {metrics['mean_confidence']:.4f}",
                transform=ax_info.transAxes, va='center', ha='left',
                fontfamily='monospace', fontsize=13, color='#FF8C00', fontweight='normal')
    
    ax_info.text(col3_x, row3_y, f"F1-Score : {metrics['f1_score']:.4f}",
                transform=ax_info.transAxes, va='center', ha='left',
                fontfamily='monospace', fontsize=13, color='#006400', fontweight='bold')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")


def evaluate_testset(checkpoint_path, testset_dir='testset', output_dir='test_evaluation',
                    threshold=0.5, iou_threshold=0.5, confidence_threshold=0.7,
                    image_size=(1080, 1920)):
    """
    Main function untuk Test evaluation
    """
    print("="*90)
    print("Test CASSITERITE DETECTION EVALUATION")
    print("="*90)
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
    model = load_model(checkpoint_path, device=device)
    
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
        
        # Load & process
        image_tensor, original_image = preprocess_image(img_path, image_size)
        gt_objects = load_ground_truth(json_path, image_size)
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
        print(f"  Pred: {metrics['total_pred']} | GT: {metrics['total_gt']}")
        print(f"  ✓ Correct: {metrics['correct']} | ⚠ Low Conf: {metrics['low_confidence']}")
        print(f"  ❌ FP: {metrics['false_positive']} | FN: {metrics['false_negative']}")
        print(f"  📍 Wrong Loc: {metrics['wrong_location']} | 🏷️  Wrong Label: {metrics['wrong_label']}")
        print(f"  📊 Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | F1: {metrics['f1_score']:.3f}")
        print(f"  📈 Mean IoU: {metrics['mean_iou']:.3f}\n")
        
        # Accumulate
        for key in ['correct', 'low_confidence', 'wrong_location', 'wrong_label',
                   'false_positive', 'false_negative', 'total_pred', 'total_gt']:
            global_metrics[key] += metrics[key]
        
        all_results.append(metrics)
    
    # Print global summary
    print("\n" + "="*90)
    print("GLOBAL SUMMARY")
    print("="*90)
    print(f"Total Images: {len(all_results)}")
    print(f"Total Predictions: {global_metrics['total_pred']}")
    print(f"Total Ground Truth: {global_metrics['total_gt']}")
    print(f"\n📊 Detection Breakdown:")
    print(f"  ✅ Correct: {global_metrics['correct']}")
    print(f"  ⚠️  Low Confidence: {global_metrics['low_confidence']}")
    print(f"  ❌ False Positives: {global_metrics['false_positive']}")
    print(f"  ❌ False Negatives: {global_metrics['false_negative']}")
    print(f"  📍 Wrong Location: {global_metrics['wrong_location']}")
    print(f"  🏷️  Wrong Label: {global_metrics['wrong_label']}")
    
    tp = global_metrics['correct'] + global_metrics['low_confidence']
    precision = tp / global_metrics['total_pred'] if global_metrics['total_pred'] > 0 else 0
    recall = tp / global_metrics['total_gt'] if global_metrics['total_gt'] > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📈 Overall Performance:")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score: {f1:.4f}")
    
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
    parser.add_argument('--image_size', type=int, nargs=2, default=[1080, 1920],
                       help='Target image size (height width)')
    
    args = parser.parse_args()
    
    evaluate_testset(
        checkpoint_path=args.checkpoint,
        testset_dir=args.testset_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        iou_threshold=args.iou_threshold,
        confidence_threshold=args.confidence_threshold,
        image_size=tuple(args.image_size)
    )


if __name__ == '__main__':
    main()
