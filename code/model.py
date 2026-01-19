import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign


def get_maskrcnn_resnet50_fpn_v1(num_classes=2, trainable_layers=3):
    """
    Create Mask R-CNN with ResNet50 + FPN V1 backbone pretrained on COCO.
    
    Args:
        num_classes: Number of classes (including background)
        trainable_layers: Number of trainable layers (0-5)
    
    Returns:
        model: Mask R-CNN V1 with COCO pretrained weights
    """
    # Load pretrained Mask R-CNN V1 with COCO weights
    model = maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1,
        trainable_backbone_layers=trainable_layers
    )
    
    # Replace box predictor for custom num_classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace mask predictor for custom num_classes
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
    return model


def get_maskrcnn_resnet50_fpn_v2(num_classes=2, trainable_layers=3):
    """
    Create Mask R-CNN with ResNet50 + FPN V2 backbone pretrained on COCO.
    V2 with COCO weights - BEST performance for transfer learning.
    
    Args:
        num_classes: Number of classes (including background)
        trainable_layers: Number of trainable layers (0-5)
    
    Returns:
        model: Mask R-CNN V2 with COCO pretrained weights
    """
    # Load pretrained Mask R-CNN V2 with COCO weights
    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
        trainable_backbone_layers=trainable_layers
    )
    
    # Replace box predictor for custom num_classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace mask predictor for custom num_classes
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
    return model

# =========================================================
#   1. Amodal Mask Head (NEW BRANCH)
# =========================================================
class AmodalMaskHead(nn.Module):
    """
    Amodal mask head:
    - 4×Conv 3x3 untuk detail lokal
    - 1×Dilated Conv untuk receptive field lebih luas
    """
    def __init__(self, in_channels=256, dilation=2):
        super().__init__()
        hidden = 256

        # 4×Conv 3x3 biasa
        convs = []
        for _ in range(4):
            convs.append(nn.Conv2d(hidden, hidden, kernel_size=3, padding=1))
            convs.append(nn.ReLU(inplace=True))
        self.local_convs = nn.Sequential(*convs)

        # Dilated Conv
        self.dilated_conv = nn.Conv2d(
            hidden, hidden,
            kernel_size=3,
            padding=dilation,
            dilation=dilation
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.local_convs(x)
        x = self.dilated_conv(x)
        x = self.relu(x)
        return x

# =========================================================
#   2. Amodal Mask Predictor
# =========================================================
class AmodalMaskPredictor(nn.Module):
    """
    Predictor terakhir untuk menghasilkan amodal mask logits
    """
    def __init__(self, in_channels=256, num_classes=2):
        super().__init__()
        self.conv_predictor = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv_predictor(x)


# =========================================================
#   3. Get Standard Mask R-CNN (ResNet50-FPN)
# =========================================================
def get_maskrcnn(backbone="v2", num_classes=2, trainable_layers=3):
    if backbone == "v1":
        model = maskrcnn_resnet50_fpn(
            weights="DEFAULT",
            trainable_backbone_layers=trainable_layers
        )
    else:
        model = maskrcnn_resnet50_fpn_v2(
            weights="DEFAULT",
            trainable_backbone_layers=trainable_layers
        )

    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace modal mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        256,
        num_classes
    )

    return model


# =========================================================
#   4. FULL AMODAL Mask R-CNN MODEL
# =========================================================
class AmodalMaskRCNN(nn.Module):
    """
    Amodal Mask R-CNN: Adds amodal mask prediction branch to standard Mask R-CNN.
    
    Architecture:
    - Backbone: ResNet50-FPN (V1 or V2)
    - RPN: Region Proposal Network
    - ROI Heads: 3 parallel branches
        1. Box regression (bounding boxes)
        2. Classification (object categories)
        3. Modal mask prediction (standard Mask R-CNN)
        4. Amodal mask prediction (NEW - predicts complete object masks)
    """
    def __init__(self, num_classes=2, backbone="v2", trainable_layers=3):
        super().__init__()

        # Base Mask R-CNN (provides backbone, RPN, and ROI heads)
        if backbone == "v1":
            self.base_model = maskrcnn_resnet50_fpn(
                weights="DEFAULT",
                trainable_backbone_layers=trainable_layers
            )
        else:  # v2
            self.base_model = maskrcnn_resnet50_fpn_v2(
                weights="DEFAULT",
                trainable_backbone_layers=trainable_layers
            )
        
        # Replace box predictor for custom num_classes
        in_features = self.base_model.roi_heads.box_predictor.cls_score.in_features
        self.base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Replace modal mask predictor for custom num_classes
        in_features_mask = self.base_model.roi_heads.mask_predictor.conv5_mask.in_channels
        self.base_model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            256,
            num_classes
        )

        # NEW: Add Amodal Mask Branch (parallel to modal mask branch)
        self.amodal_head = AmodalMaskHead(in_channels=256, dilation=2)

        # Tambahkan upsample agar output menjadi 28×28 seperti mask bawaan
        self.amodal_upsample = nn.Upsample(
            scale_factor=2, 
            mode="bilinear", 
            align_corners=False
        )

        self.amodal_predictor = AmodalMaskPredictor(in_channels=256, num_classes=num_classes)
        
        # RoI Align for amodal branch (same as modal mask branch)
        self.amodal_roi_align = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=14,
            sampling_ratio=2
        )
        
        self.num_classes = num_classes

    def forward(self, images, targets=None):
        """
        Forward pass with amodal mask prediction.
        
        Args:
            images: list of tensors [C, H, W] or ImageList
            targets: list of dicts with keys:
                - boxes: [N, 4]
                - labels: [N]
                - masks: [N, H, W] (used for BOTH modal and amodal supervision)
        
        Returns:
            Training: dict of losses including 'loss_amodal_mask'
            Inference: list of dicts with predictions including 'amodal_masks'
        """
        if self.training:
            assert targets is not None, "Targets required during training"
            
            # 1. Get standard Mask R-CNN losses (box, class, modal mask)
            losses = self.base_model(images, targets)
            
            # 2. Extract features and proposals for amodal branch
            # Convert images to ImageList if needed
            from torchvision.models.detection.image_list import ImageList
            if not isinstance(images, ImageList):
                image_sizes = [img.shape[-2:] for img in images]
                images_tensor = torch.stack(images) if isinstance(images, list) else images
                images = ImageList(images_tensor, image_sizes)
            
            # Get backbone features
            features = self.base_model.backbone(images.tensors)
            
            # Get proposals from targets (ground truth boxes during training)
            proposals = [t["boxes"] for t in targets]
            
            # 3. RoI Align for amodal branch
            amodal_features = self.amodal_roi_align(
                features, proposals, images.image_sizes
            )
            
            # 4. Amodal mask head + dilated conv
            amodal_features = self.amodal_head(amodal_features)

            # 5. Upsample 14×14 → 28×28
            amodal_features = self.amodal_upsample(amodal_features)

            # 6. Predictor
            amodal_logits = self.amodal_predictor(amodal_features)

            
            # 5. Compute amodal mask loss
            # Get ground truth amodal masks
            gt_masks = [t["masks"] for t in targets]
            gt_labels = [t["labels"] for t in targets]
            
            # Combine all masks and labels
            gt_masks_cat = torch.cat(gt_masks, dim=0).float()
            gt_labels_cat = torch.cat(gt_labels, dim=0)
            
            # IMPORTANT: Match number of GT instances to ROI features
            # RoI Align may produce fewer outputs than GT boxes (e.g., invalid boxes filtered)
            num_roi_features = amodal_logits.size(0)
            num_gt_instances = gt_labels_cat.size(0)
            
            # Truncate or pad GT to match ROI features
            if num_roi_features < num_gt_instances:
                # More GT than predictions - truncate GT
                gt_masks_cat = gt_masks_cat[:num_roi_features]
                gt_labels_cat = gt_labels_cat[:num_roi_features]
            elif num_roi_features > num_gt_instances:
                # More predictions than GT - should not happen, but handle it
                # Truncate predictions instead
                amodal_logits = amodal_logits[:num_gt_instances]
            
            # Now they should match
            num_instances = min(num_roi_features, num_gt_instances)
            
            # Ensure truncation is applied
            gt_masks_cat = gt_masks_cat[:num_instances]
            gt_labels_cat = gt_labels_cat[:num_instances]
            amodal_logits = amodal_logits[:num_instances]
            
            # Resize GT masks to match prediction size (28x28)
            gt_masks_resized = F.interpolate(
                gt_masks_cat.unsqueeze(1),
                size=(28, 28),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
            
            # Select predictions based on GT labels (same as modal mask branch)
            amodal_logits_selected = amodal_logits[torch.arange(num_instances, device=amodal_logits.device), gt_labels_cat]
            
            # Ensure shapes match before loss calculation
            assert amodal_logits_selected.shape == gt_masks_resized.shape, \
                f"Shape mismatch: pred {amodal_logits_selected.shape} vs gt {gt_masks_resized.shape}"
            
            # Binary cross-entropy loss
            loss_amodal_mask = F.binary_cross_entropy_with_logits(
                amodal_logits_selected,
                gt_masks_resized
            )
            
            # Add amodal loss to total losses
            losses['loss_amodal_mask'] = loss_amodal_mask
            
            return losses
        
        else:
            # Inference mode
            detections = self.base_model(images)
            
            # Skip if no detections or prepare for amodal predictions
            if len(detections) == 0 or all(len(d['boxes']) == 0 for d in detections):
                # No detections, add empty amodal masks
                for det in detections:
                    det['amodal_masks'] = torch.zeros(
                        (0, 1, 28, 28),
                        dtype=torch.float32,
                        device=det['boxes'].device if 'boxes' in det else 'cpu'
                    )
                return detections
            
            # Prepare images as ImageList if needed
            from torchvision.models.detection.image_list import ImageList
            if not isinstance(images, ImageList):
                if isinstance(images, list):
                    image_sizes = [img.shape[-2:] for img in images]
                    images_tensor = torch.stack(images)
                else:
                    images_tensor = images
                    image_sizes = [images.shape[-2:]]
                images = ImageList(images_tensor, image_sizes)
            
            # Get backbone features
            features = self.base_model.backbone(images.tensors)
            
            # For each detection, get amodal masks
            for i, det in enumerate(detections):
                if len(det['boxes']) == 0:
                    det['amodal_masks'] = torch.zeros(
                        (0, 1, 28, 28),
                        dtype=torch.float32,
                        device=det['boxes'].device
                    )
                    continue
                
                # RoI Align for amodal branch
                boxes_list = [det['boxes']]
                amodal_features = self.amodal_roi_align(
                    features, boxes_list, [images.image_sizes[i]]
                )
                
                # Amodal prediction
                amodal_features = self.amodal_head(amodal_features)
                amodal_features = self.amodal_upsample(amodal_features)
                amodal_logits = self.amodal_predictor(amodal_features)

                
                # Select predictions based on predicted labels
                labels = det['labels']
                num_instances = labels.size(0)
                amodal_masks = amodal_logits[torch.arange(num_instances), labels]
                
                # Apply sigmoid and add channel dim
                amodal_masks = torch.sigmoid(amodal_masks).unsqueeze(1)
                
                # Add to detection dict
                det['amodal_masks'] = amodal_masks

            return detections


# =========================================================
#   5. Factory Function
# =========================================================
def get_amodal_model(num_classes=2, backbone="v2", trainable_layers=3):
    return AmodalMaskRCNN(
        num_classes=num_classes,
        backbone=backbone,
        trainable_layers=trainable_layers
    )


def get_model(num_classes=2, model_type='standard', backbone='resnet50_fpn_v2', trainable_layers=3):
    """
    Factory function untuk membuat model sesuai Tabel 3.3.
    
    Args:
        num_classes: Jumlah kelas (default 2: background + cassiterite)
        model_type: 'standard' atau 'amodal'
        backbone: 'resnet50_fpn_v1' atau 'resnet50_fpn_v2'
        trainable_layers: Number of trainable layers in backbone (0-5)
    
    Returns:
        model: Mask R-CNN model (COCO pretrained)
        
    Model Combinations sesuai Tabel 3.3 (8 total eksperimen):
        Standard + V1:
            - SGD optimizer
            - Adam optimizer
        Standard + V2:
            - SGD optimizer
            - Adam optimizer
        Amodal + V1:
            - SGD optimizer
            - Adam optimizer
        Amodal + V2:
            - SGD optimizer
            - Adam optimizer
    """
    print(f"\nCreating Model:")
    print(f"  Type: {model_type.upper()}")
    print(f"  Backbone: {backbone.upper()}")
    print(f"  Num classes: {num_classes}")
    print(f"  Trainable layers: {trainable_layers}")
    
    if model_type == 'standard':
        # Standard Mask R-CNN with COCO pretrained
        if backbone == 'resnet50_fpn_v1':
            print(f"Creating Standard Mask R-CNN: ResNet50-FPN-V1 + COCO pretrained")
            print(f"  Trainable layers: {trainable_layers}")
            model = get_maskrcnn_resnet50_fpn_v1(
                num_classes=num_classes,
                trainable_layers=trainable_layers
            )
        elif backbone == 'resnet50_fpn_v2':
            print(f"Creating Standard Mask R-CNN: ResNet50-FPN-V2 + COCO pretrained")
            print(f"  Trainable layers: {trainable_layers}")
            model = get_maskrcnn_resnet50_fpn_v2(
                num_classes=num_classes,
                trainable_layers=trainable_layers
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}. Use 'resnet50_fpn_v1' or 'resnet50_fpn_v2'")
    
    elif model_type == 'amodal':
        # Amodal Mask R-CNN with COCO pretrained
        print(f"  Architecture: Mask R-CNN + Amodal Branch")
        
        # Extract backbone version (v1 or v2)
        backbone_version = "v1" if "v1" in backbone else "v2"
        
        model = AmodalMaskRCNN(
            num_classes=num_classes,
            backbone=backbone_version,
            trainable_layers=trainable_layers
        )
        
        print(f"  ✓ Amodal branch added (parallel mask prediction)")
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'standard' or 'amodal'")
    
    return model


def count_parameters(model):
    """
    Count total dan trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params

if __name__ == '__main__':
    # Test model creation
    print("="*80)
    print("TESTING ALL MODELS (COCO Pretrained)")
    print("="*80)
    
    print("\n1. Standard Mask R-CNN with ResNet50-FPN-V1...")
    model1 = get_model(num_classes=2, model_type='standard', backbone='resnet50_fpn_v1')
    count_parameters(model1)
    
    print("\n2. Standard Mask R-CNN with ResNet50-FPN-V2...")
    model2 = get_model(num_classes=2, model_type='standard', backbone='resnet50_fpn_v2')
    count_parameters(model2)
    
    print("\n3. Amodal Mask R-CNN with ResNet50-FPN-V1...")
    model3 = get_model(num_classes=2, model_type='amodal', backbone='resnet50_fpn_v1')
    count_parameters(model3)
    
    print("\n4. Amodal Mask R-CNN with ResNet50-FPN-V2...")
    model4 = get_model(num_classes=2, model_type='amodal', backbone='resnet50_fpn_v2')
    count_parameters(model4)
    
    print("\n" + "="*80)
    print("Model creation successful!")
    print("All 4 model variants created successfully (all COCO pretrained)")
    print("="*80)


