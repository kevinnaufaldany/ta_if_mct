uploud data to runpod

```bash
# Local PC
tar -czf dataset142.tar.gz dataset142

# Upload via RunPod Web UI → Files tab

# RunPod Terminal
cd /workspace
```
```bash
tar -xzf dataset142.tar.gz
```
```bash
python testjson.py dataset142 
``` 


- Variasi : 
```py
{config['backbone']} + {config['model_type']} + {config['optimizer']}
```

1. maskrcnn_resnet50_fpn + Amodal + SGD
```bash
python train.py --data_dir dataset142 --backbone resnet50_fpn_v1 --model_type amodal --optimizer sgd --lr 1e-2 --batch_size 8
```

2. maskrcnn_resnet50_fpn + Standar + Adam
```bash
python train.py --data_dir dataset142 --backbone resnet50_fpn_v1 --model_type standard --optimizer adam --lr 1e-4 --batch_size 8
```

3. maskrcnn_resnet50_fpn + Amodal + SGD
```bash
python train.py --data_dir dataset142 --backbone resnet50_fpn_v1 --model_type amodal --optimizer sgd --lr 1e-2 --batch_size 8
```

4. maskrcnn_resnet50_fpn + Standar + Adam
```bash
python train.py --data_dir dataset142 --backbone resnet50_fpn_v1 --model_type standard  --optimizer adam --lr 1e-4 --batch_size 8
```

5. maskrcnn_resnet50_fpn_v2 + Amodal + SGD
```bash
python train.py --data_dir dataset142 --backbone resnet50_fpn_v2 --model_type amodal --optimizer sgd --lr 1e-2 --batch_size 8
```

6. maskrcnn_resnet50_fpn_v2 + Amodal + Adam
```bash
python train.py --data_dir dataset142 --backbone resnet50_fpn_v2 --model_type standard --optimizer adam --lr 1e-4 --batch_size 8
```

7. maskrcnn_resnet50_fpn_v2 + Standar + SGD
```bash
python train.py --data_dir dataset142 --backbone resnet50_fpn_v2 --model_type standard  --optimizer sgd --lr 1e-2 --batch_size 8
```

8. maskrcnn_resnet50_fpn_v2 + Standar + Adam
```bash
python train.py --data_dir dataset142 --backbone resnet50_fpn_v2 --model_type standard --optimizer adam --lr 1e-4 --batch_size 8
```


opsional 
9. maskrcnn_resnet50_fpn + Standar + Adam
```bash
python train.py --data_dir dataset142 --backbone resnet50_fpn_v1 --model_type standard --optimizer adam --lr 1e-4 --batch_size 8
```

10. maskrcnn_resnet50_fpn + Standar + Adam
```bash
python train.py --data_dir dataset142 --backbone resnet50_fpn_v1 --model_type standard  --optimizer adam --lr 1e-4 --batch_size 8
```

11. maskrcnn_resnet50_fpn_v2 + Amodal + Adam
```bash
python train.py --data_dir dataset142 --backbone resnet50_fpn_v2 --model_type standard --optimizer adam --lr 1e-4 --batch_size 8
```

12. maskrcnn_resnet50_fpn_v2 + Standar + Adam
```bash
python train.py --data_dir dataset142 --backbone resnet50_fpn_v2 --model_type standard --optimizer adam --lr 1e-4 --batch_size 8
```