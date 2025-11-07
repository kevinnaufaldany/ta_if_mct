uploud data to runpod

```bash
# Local PC
tar -czf dataset11.tar.gz dataset11

# Upload via RunPod Web UI → Files tab

# RunPod Terminal
cd /workspace
```
```bash
tar -xzf dataset11.tar.gz
```
```bash
python testjson.py dataset11 
``` 


- Variasi : 
```py
{config['backbone']} + {config['model_type']} + {config['optimizer']}
```

1. maskrcnn_resnet50_fpn + Amodal + SGD
```bash
python train.py --backbone resnet50_fpn_v1 --model_type amodal --optimizer sgd --lr 1e-2
```

2. maskrcnn_resnet50_fpn + Amodal + Adam
```bash
python train.py --backbone resnet50_fpn_v1 --model_type amodal --optimizer adam --lr 1e-4
```

3. maskrcnn_resnet50_fpn + Standar + SGD
```bash
python train.py --backbone resnet50_fpn_v1 --model_type standard --optimizer sgd --lr 1e-2
```

4. maskrcnn_resnet50_fpn + Standar + Adam
```bash
python train.py --backbone resnet50_fpn_v1 --model_type standard  --optimizer adam --lr 1e-4
```

5. maskrcnn_resnet50_fpn_v2 + Amodal + SGD
```bash
python train.py --backbone resnet50_fpn_v2 --model_type amodal --optimizer sgd --lr 1e-2
```

6. maskrcnn_resnet50_fpn_v2 + Amodal + Adam
```bash
python train.py --backbone resnet50_fpn_v2 --model_type standard --optimizer adam --lr 1e-4
```

7. maskrcnn_resnet50_fpn_v2 + Standar + SGD
```bash
python train.py --backbone resnet50_fpn_v2 --model_type standard  --optimizer sgd --lr 1e-2
```

8. maskrcnn_resnet50_fpn_v2 + Standar + Adam
```bash
python train.py --backbone resnet50_fpn_v2 --model_type standard --optimizer adam --lr 1e-4
```


opsional 
9. maskrcnn_resnet50_fpn + Standar + Adam
```bash
python train.py --backbone resnet50_fpn_v1 --model_type standard --optimizer adam --lr 1e-4
```

10. maskrcnn_resnet50_fpn + Standar + Adam
```bash
python train.py --backbone resnet50_fpn_v1 --model_type standard  --optimizer adam --lr 1e-4
```

11. maskrcnn_resnet50_fpn_v2 + Amodal + Adam
```bash
python train.py --backbone resnet50_fpn_v2 --model_type standard --optimizer adam --lr 1e-4
```

12. maskrcnn_resnet50_fpn_v2 + Standar + Adam
```bash
python train.py --backbone resnet50_fpn_v2 --model_type standard --optimizer adam --lr 1e-4
```