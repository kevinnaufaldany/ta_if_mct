import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from sklearn.model_selection import KFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

ROOT_DIR = 'dataset11'  

class CassiteriteDataset(Dataset):
    """
    Dataset untuk amodal instance segmentation pada mineral cassiterite 5-fold cross-validation / augmentasi tipis.
    """
    def __init__(self, root_dir, image_files, target_size=(1080, 1920), transform=None, is_train=True):
        self.root_dir = root_dir
        self.image_files = image_files
        self.target_size = target_size
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = cv2.imread(img_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotation
        json_name = os.path.splitext(img_name)[0] + '.json'
        json_path = os.path.join(self.root_dir, json_name)
        
        try:
            # Try to read with UTF-8 encoding first
            with open(json_path, 'r', encoding='utf-8') as f:
                content = f.read()
                annotation = json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            # If UTF-8 fails, try with latin-1
            try:
                with open(json_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                    annotation = json.loads(content)
                print(f"Warning: {json_path} loaded with latin-1 encoding")
            except Exception as e2:
                # Last resort: try to fix common issues
                try:
                    with open(json_path, 'rb') as f:
                        raw_content = f.read()
                    # Remove null bytes and other problematic characters
                    clean_content = raw_content.decode('utf-8', errors='ignore')
                    annotation = json.loads(clean_content)
                    print(f"Warning: {json_path} loaded with error correction")
                except Exception as e3:
                    raise ValueError(
                        f"  JSON PARSE ERROR in {json_path}:\n"
                        f"  File: {img_name}\n"
                        f"  Error: {str(e)}\n"
                        f"  Line/Char: Check if file is corrupt or incomplete.\n"
                        f"  Solution: Re-upload this file to RunPod."
                    ) from e
        except FileNotFoundError:
            raise ValueError(f"JSON file not found: {json_path}")
        
        # Parse shapes (polygons)
        masks = []
        boxes = []
        labels = []
        
        original_h, original_w = image.shape[:2]
        
        for shape in annotation['shapes']:
            if shape['shape_type'] == 'polygon':
                points = np.array(shape['points'], dtype=np.float32)
                
                # Create mask untuk polygon
                mask = np.zeros((original_h, original_w), dtype=np.uint8)
                points_int = points.astype(np.int32)
                cv2.fillPoly(mask, [points_int], 1)
                
                # Get bounding box
                x_coords = points[:, 0]
                y_coords = points[:, 1]
                xmin = np.min(x_coords)
                ymin = np.min(y_coords)
                xmax = np.max(x_coords)
                ymax = np.max(y_coords)
                
                # Skip jika box terlalu kecil
                if (xmax - xmin) < 5 or (ymax - ymin) < 5:
                    continue
                
                masks.append(mask)
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1)  # Kelas 1 untuk cassiterite (0 adalah background)
        
        # Convert to numpy arrays
        if len(masks) == 0:
            # Jika tidak ada objek, buat dummy
            # masks = np.zeros((1, original_h, original_w), dtype=np.uint8)
            # boxes = np.array([[0, 0, 1, 1]], dtype=np.float32)
            # labels = np.array([0], dtype=np.int64)
            raise ValueError(f"No valid object found in {img_name}")

        else:
            masks = np.array(masks, dtype=np.uint8)
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
        
        # Apply transformations
        if self.transform:
            # Prepare untuk albumentations
            transformed = self.transform(
                image=image,
                masks=list(masks),
                bboxes=boxes,
                labels=labels
            )
            image = transformed['image']
            masks = np.array(transformed['masks'])
            boxes = np.array(transformed['bboxes'])
            labels = np.array(transformed['labels'])
        else:
            # Default resize tanpa augmentasi
            image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
            
            # Resize masks
            resized_masks = []
            for mask in masks:
                resized_mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), 
                                         interpolation=cv2.INTER_NEAREST)
                resized_masks.append(resized_mask)
            masks = np.array(resized_masks)
            
            # Scale boxes
            scale_x = self.target_size[1] / original_w
            scale_y = self.target_size[0] / original_h
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            
            # Convert image to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Prepare target dictionary untuk Mask R-CNN
        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        target['masks'] = torch.as_tensor(masks, dtype=torch.uint8)
        target['image_id'] = torch.tensor([idx])
        target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target['area'] = torch.as_tensor(target['area'], dtype=torch.float32)
        target['iscrowd'] = torch.zeros((len(labels),), dtype=torch.int64)
        
        return image, target


def get_train_transform(target_size=(1080, 1920)):
    """
    Augmentasi tipis untuk training - resize, flip, rotate, noise ringan.
    """
    return A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.HorizontalFlip(p=0.4),
        A.VerticalFlip(p=0.4),
        A.Rotate(limit=(-15, 15), p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.3), 
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2(), # normaliasasi /255
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
       additional_targets={'masks': 'masks'})


def get_val_transform(target_size=(1080, 1920)):
    """
    Transformasi untuk validation - hanya resize dan normalisasi.
    """
    return A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2(), # normaliasasi /255
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
       additional_targets={'masks': 'masks'})


def prepare_kfold_datasets(root_dir=ROOT_DIR, n_splits=5, target_size=(1080, 1920)):
    """
    Mempersiapkan 5-fold cross-validation datasets List of (train_dataset, val_dataset) untuk setiap fold
    """
    # Get all image files
    all_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    all_files = sorted(all_files)
    
    print(f"Total images found: {len(all_files)}")
    
    # K-Fold split
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_datasets = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_files)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        train_files = [all_files[i] for i in train_idx]
        val_files = [all_files[i] for i in val_idx]
        
        print(f"  Train: {len(train_files)} images")
        print(f"  Val: {len(val_files)} images")
        
        # Create datasets
        train_dataset = CassiteriteDataset(
            root_dir=root_dir,
            image_files=train_files,
            target_size=target_size,
            transform=get_train_transform(target_size),
            is_train=True
        )
        
        val_dataset = CassiteriteDataset(
            root_dir=root_dir,
            image_files=val_files,
            target_size=target_size,
            transform=get_val_transform(target_size),
            is_train=False
        )
        
        fold_datasets.append((train_dataset, val_dataset))
    
    return fold_datasets


def collate_fn(batch):
    """
    Custom collate function untuk Mask R-CNN.
    Karena setiap gambar bisa punya jumlah objek berbeda.
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    return images, targets


def get_dataloader(dataset, batch_size=2, shuffle=True, num_workers=4):
    """
    Create DataLoader dengan custom collate function.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
