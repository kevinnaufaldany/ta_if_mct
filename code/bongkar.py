import argparse
import torch
from model import get_model, count_parameters  # ganti 'model_file' dengan nama file aslimu
from torchsummary import summary


def print_submodule_parameters(model, name):
    """
    Cetak jumlah parameter di submodule tertentu.
    """
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== {name} ===")
    print(f"Total Params     : {params:,}")
    print(f"Trainable Params : {trainable:,}")


def inspect_model(model):
    """
    Bongkar seluruh arsitektur Mask R-CNN / Amodal Mask R-CNN.
    """
    print("\n==========================")
    print("🔍 MODEL ARCHITECTURE DUMP")
    print("==========================\n")

    print(model)   # print seluruh arsitektur PyTorch

    print("\n==========================")
    print("🔢 PARAMETER PER SUBMODULE")
    print("==========================")

    # Backbone (ResNet50-FPN)
    print_submodule_parameters(model.base_model.backbone, "Backbone (ResNet50-FPN)")

    # RPN
    print_submodule_parameters(model.base_model.rpn, "RPN (Region Proposal Network)")

    # ROI HEADS
    print_submodule_parameters(model.base_model.roi_heads.box_predictor, "ROI Box Predictor")
    print_submodule_parameters(model.base_model.roi_heads.mask_predictor, "ROI Mask Predictor (Modal Mask)")

    # Jika Amodal
    if hasattr(model, "amodal_head"):
        print_submodule_parameters(model.amodal_head, "Amodal Mask Head (4×Conv)")
        print_submodule_parameters(model.amodal_predictor, "Amodal Mask Predictor (1×1 Conv)")

    print("\n==========================")
    print("📌 TOTAL PARAMETERS MODEL")
    print("==========================")
    count_parameters(model)

    print("\nSelesai membongkar model!\n")


def main():
    parser = argparse.ArgumentParser(description="Bongkar Mask R-CNN Models Kamu")
    parser.add_argument("--model_type", type=str, default="standard",
                        choices=["standard", "amodal"],
                        help="Jenis model: standard / amodal")

    parser.add_argument("--backbone", type=str, default="v2",
                        choices=["v1", "v2"],
                        help="Backbone: v1 atau v2")

    parser.add_argument("--num_classes", type=int, default=2,
                        help="Jumlah kelas termasuk background")

    parser.add_argument("--trainable_layers", type=int, default=3,
                        help="Jumlah trainable layers Backbone")

    args = parser.parse_args()

    # Backbone format matching your get_model()
    backbone_name = (
        "resnet50_fpn_v1" if args.backbone == "v1" else "resnet50_fpn_v2"
    )

    print("\n=========================================")
    print(f"  LOADING MODEL: {args.model_type.upper()} + {args.backbone.upper()}")
    print("=========================================\n")

    model = get_model(
        num_classes=args.num_classes,
        model_type=args.model_type,
        backbone=backbone_name,
        trainable_layers=args.trainable_layers
    )

    model.eval()

    # Bongkar model
    inspect_model(model)


if __name__ == "__main__":
    main()
