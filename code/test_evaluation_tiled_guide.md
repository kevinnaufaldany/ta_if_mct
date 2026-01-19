# Test Evaluation Guide - 8 Variasi Model

```bash
python test_evaluation_tiled.py --checkpoint "resnet50_fpn_v2_amodal_adam_lr1e-04\fold1\best_ap_epoch15.pth" --model_type amodal --backbone resnet50_fpn_v2 --testset_dir testset --output_dir test_evaluation2/v2_amodal_adam8
```

1. maskrcnn_resnet50_fpn + Amodal + SGD
```bash
python test_evaluation_tiled.py --checkpoint "resnet50_fpn_v2_amodal_adam_lr1e-04\fold1\best_ap_epoch15.pth" --model_type amodal --backbone resnet50_fpn_v2 --testset_dir testset --output_dir test_evaluation/v1_amodal_sgd
```

2. maskrcnn_resnet50_fpn + Amodal + Adam
```bash
python test_evaluation_tiled.py --checkpoint "resnet50_fpn_v2_amodal_adam_lr1e-04\fold1\best_ap_epoch15.pth" --model_type amodal --backbone resnet50_fpn_v2 --testset_dir testset --output_dir test_evaluation/v1_standard_adam
```

3. maskrcnn_resnet50_fpn + Standard + SGD
```bash
python test_evaluation_tiled.py --checkpoint "resnet50_fpn_v2_amodal_adam_lr1e-04\fold1\best_ap_epoch15.pth" --model_type amodal --backbone resnet50_fpn_v2 --testset_dir testset --output_dir test_evaluation/v1_amodal_adam
```

4. maskrcnn_resnet50_fpn + Standard + Adam
```bash
python test_evaluation_tiled.py --checkpoint "resnet50_fpn_v2_amodal_adam_lr1e-04\fold1\best_ap_epoch15.pth" --model_type amodal --backbone resnet50_fpn_v2 --testset_dir testset --output_dir test_evaluation/v1_standard_sgd
```

5. maskrcnn_resnet50_fpn_v2 + Amodal + SGD
```bash
python test_evaluation_tiled.py --checkpoint "resnet50_fpn_v2_amodal_adam_lr1e-04\fold1\best_ap_epoch15.pth" --model_type amodal --backbone resnet50_fpn_v2 --testset_dir testset --output_dir test_evaluation/v2_amodal_sgd
```

6. maskrcnn_resnet50_fpn_v2 + Amodal + Adam
```bash
python test_evaluation_tiled.py --checkpoint "resnet50_fpn_v2_amodal_adam_lr1e-04\fold1\best_ap_epoch15.pth" --model_type amodal --backbone resnet50_fpn_v2 --testset_dir testset --output_dir test_evaluation/v2_standard_adam
```

7. maskrcnn_resnet50_fpn_v2 + Standard + SGD
```bash
python test_evaluation_tiled.py --checkpoint "resnet50_fpn_v2_amodal_adam_lr1e-04\fold1\best_ap_epoch15.pth" --model_type amodal --backbone resnet50_fpn_v2 --testset_dir testset --output_dir test_evaluation/v2_standard_sgd
```

8. maskrcnn_resnet50_fpn_v2 + Standard + Adam
```bash
python test_evaluation_tiled.py --checkpoint "resnet50_fpn_v2_amodal_adam_lr1e-04\fold1\best_ap_epoch15.pth" --model_type amodal --backbone resnet50_fpn_v2 --testset_dir testset --output_dir test_evaluation/v2_amodal_adam
```

