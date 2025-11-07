import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_maskrcnn_resnet50_fpn_v1(num_classes=2, trainable_layers=3):
    """
    Create Mask R-CNN with ResNet50 + FPN V1 backbone pretrained on COCO.
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


class AmodalMaskRCNN(nn.Module):
    """
    Amodal Mask R-CNN wrapper for complete object segmentation.
    Supports ResNet50 FPN V1 and V2 with COCO pretrained weights.
    All models use COCO pretrained weights for best transfer learning.
    """
    def __init__(self, num_classes=2, backbone='resnet50_fpn_v2', trainable_layers=3):
        super(AmodalMaskRCNN, self).__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Create base Mask R-CNN model
        if backbone == 'resnet50_fpn_v1':
            print(f"Creating Amodal Mask R-CNN: ResNet50-FPN-V1 + COCO pretrained")
            self.maskrcnn = get_maskrcnn_resnet50_fpn_v1(
                num_classes=num_classes,
                trainable_layers=trainable_layers
            )
                
        elif backbone == 'resnet50_fpn_v2':
            print(f"Creating Amodal Mask R-CNN: ResNet50-FPN-V2 + COCO pretrained")
            self.maskrcnn = get_maskrcnn_resnet50_fpn_v2(
                num_classes=num_classes,
                trainable_layers=trainable_layers
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}. Use 'resnet50_fpn_v1' or 'resnet50_fpn_v2'")
    
    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        # Forward through Mask R-CNN
        if self.training:
            # Training mode - return losses
            loss_dict = self.maskrcnn(images, targets)
            return loss_dict
        else:
            # Inference mode - return predictions
            detections = self.maskrcnn(images)
            
            # Post-process untuk amodal segmentation
            detections = self.postprocess_amodal(detections)
            
            return detections
    
    def postprocess_amodal(self, detections):
        """
        Post-processing untuk amodal predictions, still a placeholder.
        """
        return detections

def get_model(num_classes=2, model_type='standard', backbone='resnet50_fpn_v2', trainable_layers=3):
    """
    Factory function untuk membuat model.
    """
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
        print(f"  Trainable layers: {trainable_layers}")
        model = AmodalMaskRCNN(
            num_classes=num_classes,
            backbone=backbone,
            trainable_layers=trainable_layers
        )
    
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


