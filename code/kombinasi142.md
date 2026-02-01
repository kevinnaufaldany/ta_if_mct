uploud data to runpod

```bash
# zipping
tar -czf hasil_train_all.tar.gz hasil_train_all

# extract
tar -xzf hasil_train_all.tar.gz 

# Download dataset
wget -c https://github.com/kevinnaufaldany/ta_if_mct/releases/download/data/dataset142.tar.gz


# RunPod Terminal
cd /workspace
```
```bash
tar -xzf dataset142.tar.gz
```
```bash
pip install -r requirements.txt
```
```bash
wandb login

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

```
```bash
rclone version
``` 
```bash
sudo apt update
``` 
```bash
sudo apt install rclone -y
``` 
```bash
rclone config
``` 

```bash
n) New remote
name> gdrive1
``` 
```bash
Storage> drive
``` 
```bash
client_id>        [Enter]
client_secret>   [Enter]
``` 
```bash
scope> 1
``` 
```bash
root_folder_id>      [Enter]
service_account_file> [Enter]
``` 
```bash
Edit advanced config? n
``` 

Set up di Terminal PC Lokal 
```bash
rclone authorize "drive" "[Token Runpod]"
``` 
```bash
config_token> [PASTE JSON TOKEN FROM TERMINAL PC LOKAL]
``` 
```bash
Configure this as a Shared Drive? n
``` 
```bash
y) Yes this is OK
``` 

```bash
rclone lsd gdrive1:
``` 
```bash
tar -czvf hasil_train_all.tar.gz output checkpoints
``` 
```bash
rclone copy hasil_train_all.tar.gz gdrive1:Kevin/runpod --progress
``` 
```bash
rclone ls gdrive1:Kevin/runpod
``` 



- Variasi : 
```py
{config['backbone']} + {config['model_type']} + {config['optimizer']}
```

1. maskrcnn_resnet50_fpn + Amodal + SGD
```bash
python train.py --data_dir dataset142 --backbone resnet50_fpn_v1 --model_type amodal --optimizer sgd --lr 1e-2 --batch_size 1 --no_amp --num_workers 14 --COSINE_ETA_MIN 1e-4
```

2. maskrcnn_resnet50_fpn + Amodal + Adam
```bash
python train.py --data_dir dataset142 --backbone resnet50_fpn_v1 --model_type amodal --optimizer adam --lr 1e-4 --batch_size 1 --weight_decay 1e-5 --no_amp --num_workers 14 
```

3. maskrcnn_resnet50_fpn + Standard + SGD
```bash
python train.py --data_dir dataset142 --backbone resnet50_fpn_v1 --model_type standard --optimizer sgd --lr 1e-2 --batch_size 1 --no_amp --num_workers 14 --COSINE_ETA_MIN 1e-4
```

4. maskrcnn_resnet50_fpn + Standard + Adam
```bash
python train.py --data_dir dataset142 --backbone resnet50_fpn_v1 --model_type standard --optimizer adam --lr 1e-4 --batch_size 1 --weight_decay 1e-5 --no_amp --num_workers 14 
```

5. maskrcnn_resnet50_fpn_v2 + Amodal + SGD
```bash
python train.py --data_dir dataset142 --backbone resnet50_fpn_v2 --model_type amodal --optimizer sgd --lr 1e-2 --batch_size 1 --no_amp --num_workers 14 --COSINE_ETA_MIN 1e-4
```

6. maskrcnn_resnet50_fpn_v2 + Amodal + Adam
```bash
python train.py --data_dir dataset142 --backbone resnet50_fpn_v2 --model_type amodal --optimizer adam --lr 1e-4 --batch_size 1 --weight_decay 1e-5 --no_amp --num_workers 14 
```

7. maskrcnn_resnet50_fpn_v2 + Standard + SGD
```bash
python train.py --data_dir dataset142 --backbone resnet50_fpn_v2 --model_type standard  --optimizer sgd --lr 1e-2 --batch_size 1 --no_amp --num_workers 14 --COSINE_ETA_MIN 1e-4
```

8. maskrcnn_resnet50_fpn_v2 + Standard + Adam
```bash
python train.py --data_dir dataset142 --backbone resnet50_fpn_v2 --model_type standard --optimizer adam --lr 1e-4 --batch_size 1 --weight_decay 1e-5 --no_amp --num_workers 14 
```