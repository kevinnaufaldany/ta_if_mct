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
from shapely.geometry import Polygon, box

ROOT_DIR = 'dataset142'  

def clip_polygon(points, crop_box):
    """
    Clip polygon dengan crop box menggunakan shapely.
    """
    try:
        poly = Polygon(points)
        
        # Fix invalid geometry
        if not poly.is_valid:
            poly = poly.buffer(0)
        
        # Ensure crop_box is valid too
        if not crop_box.is_valid:
            crop_box = crop_box.buffer(0)
        
        clipped = poly.intersection(crop_box)
        
        if clipped.is_empty:
            return None
        if clipped.geom_type == "Polygon":
            return np.array(clipped.exterior.coords)
        elif clipped.geom_type == "MultiPolygon":
            # Ambil polygon terbesar jika hasil intersection adalah MultiPolygon
            largest = max(clipped.geoms, key=lambda p: p.area)
            return np.array(largest.exterior.coords)
        return None
    except Exception as e:
        # Skip polygon yang error
        return None


def get_valid_tiles_from_image(root_dir, img_name, grid=(2, 2)):
    """
    Helper function untuk mendapatkan valid tiles dari satu image.
    Returns: list of (img_name, tile_id) yang memiliki objek.
    """
    img_path = os.path.join(root_dir, img_name)
    
    # Load image untuk mendapatkan dimensi
    image = cv2.imread(img_path)
    if image is None:
        return []
    
    original_h, original_w = image.shape[:2]
    
    # Load annotation
    json_name = os.path.splitext(img_name)[0] + '.json'
    json_path = os.path.join(root_dir, json_name)
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)
    except:
        return []
    
    # Calculate tile boundaries
    tile_h = original_h // grid[0]
    tile_w = original_w // grid[1]
    
    valid_tiles = []
    n_tiles = grid[0] * grid[1]
    
    for tile_id in range(n_tiles):
        i = tile_id // grid[1]
        j = tile_id % grid[1]
        
        x1 = j * tile_w
        y1 = i * tile_h
        x2 = x1 + tile_w
        y2 = y1 + tile_h
        
        crop_box = box(x1, y1, x2, y2)
        
        # Check if ada polygon yang intersect dengan tile ini
        has_object = False
        for shape in annotation.get('shapes', []):
            if shape['shape_type'] == 'polygon':
                points = np.array(shape['points'], dtype=np.float32)
                clipped = clip_polygon(points, crop_box)
                
                if clipped is not None:
                    # Check if resulting polygon is large enough
                    x_coords = clipped[:, 0]
                    y_coords = clipped[:, 1]
                    width = np.max(x_coords) - np.min(x_coords)
                    height = np.max(y_coords) - np.min(y_coords)
                    
                    if width >= 5 and height >= 5:
                        has_object = True
                        break
        
        if has_object:
            valid_tiles.append((img_name, tile_id))
    
    return valid_tiles


class CassiteriteDataset(Dataset):
    """
    Dataset untuk amodal instance segmentation pada mineral cassiterite dengan automatic splitting.
    Data akan di-split menjadi tiles 2x2 terlebih dahulu sebelum augmentasi.
    Hanya tiles yang memiliki objek yang akan dimasukkan ke dataset.
    """
    def __init__(self, root_dir, tile_list, grid=(2, 2), transform=None, is_train=True):
        self.root_dir = root_dir
        self.tile_index = tile_list  # List of (img_name, tile_id)
        self.grid = grid  # Grid untuk splitting (default 2x2)
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.tile_index)
    
    def __getitem__(self, idx):
        # Get image file and tile index
        img_name, tile_id = self.tile_index[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        # Load full image
        image = cv2.imread(img_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]
        
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
        
        # Calculate tile boundaries
        tile_h = original_h // self.grid[0]
        tile_w = original_w // self.grid[1]
        
        i = tile_id // self.grid[1]
        j = tile_id % self.grid[1]
        
        x1 = j * tile_w
        y1 = i * tile_h
        x2 = x1 + tile_w
        y2 = y1 + tile_h
        
        # Crop image to tile
        image = image[y1:y2, x1:x2]
        crop_box = box(x1, y1, x2, y2)
        
        # Parse shapes (polygons) and clip to tile
        masks = []
        boxes = []
        labels = []
        
        for shape in annotation['shapes']:
            if shape['shape_type'] == 'polygon':
                points = np.array(shape['points'], dtype=np.float32)
                
                # Clip polygon to tile boundaries
                clipped = clip_polygon(points, crop_box)
                if clipped is None:
                    continue
                
                # Shift coordinates to local tile coordinates
                clipped[:, 0] -= x1
                clipped[:, 1] -= y1
                
                # Create mask untuk clipped polygon (dalam koordinat tile)
                tile_h_actual, tile_w_actual = image.shape[:2]
                mask = np.zeros((tile_h_actual, tile_w_actual), dtype=np.uint8)
                points_int = clipped.astype(np.int32)
                cv2.fillPoly(mask, [points_int], 1)
                
                # Get bounding box (dalam koordinat tile)
                x_coords = clipped[:, 0]
                y_coords = clipped[:, 1]
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
            # print(f"Warning: Tile {tile_id} of {img_name} has no objects. Skipping to next...")
            return self.__getitem__((idx + 1) % len(self.tile_index))
        
        masks = np.array(masks, dtype=np.uint8)
        labels = np.array(labels, dtype=np.int64)
        
        # Apply transformations
        if self.transform:
            # Prepare untuk albumentations - hanya pass masks, tidak bboxes
            masks_list = [m for m in masks]
            
            transformed = self.transform(
                image=image,
                masks=masks_list
            )
            image = transformed['image']
            
            # Handle transformed masks
            transformed_masks = transformed['masks']
            if len(transformed_masks) == 0:
                # print(f"Warning: Augmentation removed all objects in tile {tile_id} of {img_name}. Skipping to next...")
                return self.__getitem__((idx + 1) % len(self.tile_index))
            
            masks = np.array(transformed_masks)
            
            # Generate bboxes dari transformed masks
            boxes = []
            valid_indices = []
            for i, mask in enumerate(masks):
                # Find bounding box dari mask
                pos = np.where(mask)
                if len(pos[0]) == 0:  # Empty mask
                    continue
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                
                # Skip jika box terlalu kecil
                if (xmax - xmin) < 5 or (ymax - ymin) < 5:
                    continue
                
                boxes.append([xmin, ymin, xmax, ymax])
                valid_indices.append(i)
            
            if len(boxes) == 0:
                # print(f"Warning: No valid boxes after augmentation in tile {tile_id} of {img_name}. Skipping to next...")
                return self.__getitem__((idx + 1) % len(self.tile_index))
            
            # Filter masks dan labels berdasarkan valid indices
            masks = masks[valid_indices]
            labels = labels[valid_indices]
            boxes = np.array(boxes, dtype=np.float32)
        else:
            # Convert image to tensor tanpa augmentasi
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            boxes = np.array(boxes, dtype=np.float32)
        
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


def get_train_transform():
    """
    Augmentasi tipis untuk training - flip, rotate, noise ringan.
    Tidak ada resize, ukuran mengikuti tile size.
    Bboxes akan di-generate dari masks setelah transform.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.4),
        A.VerticalFlip(p=0.4),
        A.Rotate(limit=(-15, 15), p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.GaussNoise(std_range=(0.44721359549995793928183473374626, 0.44721359549995793928183473374626), mean_range=(0.0, 0.0), per_channel=True, p=0.3), # Gaussian noise dengan variansi 0.2
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2(),
    ])


def get_val_transform():
    """
    Transformasi untuk validation - hanya normalisasi.
    Tidak ada resize, ukuran mengikuti tile size.
    Bboxes akan di-generate dari masks setelah transform.
    """
    return A.Compose([
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2(),
    ])


def prepare_kfold_datasets(root_dir=ROOT_DIR, n_splits=5, grid=(2, 2)):
    """
    Mempersiapkan 5-fold cross-validation datasets dengan automatic splitting.
    K-Fold dilakukan pada tiles, bukan images.
    
    Flow:
    1. Scan semua images
    2. Pre-compute semua valid tiles (yang memiliki objek)
    3. Split tiles menjadi K-folds
    4. Return datasets untuk setiap fold
    """
    # Get all image files
    all_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    all_files = sorted(all_files)
    
    print(f"Total images found: {len(all_files)}")
    print(f"Grid split: {grid[0]}x{grid[1]} = {grid[0] * grid[1]} tiles per image")
    
    # Pre-compute semua valid tiles dari semua images
    print("\nPre-computing all valid tiles...")
    all_tiles = []
    for img_file in all_files:
        valid_tiles = get_valid_tiles_from_image(root_dir, img_file, grid)
        all_tiles.extend(valid_tiles)
    
    print(f"Total valid tiles: {len(all_tiles)}")
    
    # K-Fold split pada tiles, bukan images
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_datasets = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_tiles)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        train_tiles = [all_tiles[i] for i in train_idx]
        val_tiles = [all_tiles[i] for i in val_idx]
        
        print(f"  Train tiles: {len(train_tiles)}")
        print(f"  Val tiles: {len(val_tiles)}")
        
        # Create datasets dengan tile_list langsung
        train_dataset = CassiteriteDataset(
            root_dir=root_dir,
            tile_list=train_tiles,
            grid=grid,
            transform=get_train_transform(),
            is_train=True
        )
        
        val_dataset = CassiteriteDataset(
            root_dir=root_dir,
            tile_list=val_tiles,
            grid=grid,
            transform=get_val_transform(),
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

# Example usage: 
if __name__ == "__main__":
    # Prepare datasets
    fold_datasets = prepare_kfold_datasets(root_dir=ROOT_DIR, n_splits=5, grid=(2, 2))
    
    # Ambil fold pertama untuk testing
    train_dataset, val_dataset = fold_datasets[0]
    
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    # 1. Data Original (sebelum tile)
    all_files = [f for f in os.listdir(ROOT_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\n1. DATA ORIGINAL:")
    print(f"   Total images: {len(all_files)}")
    
    # 2. Data After Tile
    grid = (2, 2)
    total_valid_tiles = len(train_dataset) + len(val_dataset)
    print(f"\n2. DATA AFTER TILE ({grid[0]}x{grid[1]}):")
    print(f"   Total valid tiles: {total_valid_tiles}")
    print(f"   Train tiles: {len(train_dataset)}")
    print(f"   Val tiles: {len(val_dataset)}")
    
    # 3. Sample augmentasi
    print(f"\n3. SAMPLE AUGMENTASI:")
    print(f"   Loading sample from train dataset...")
    
    try:
        # Ambil 1 sample dari train
        image, target = train_dataset[0]
        
        print(f"   Image shape: {image.shape}")
        print(f"   Number of objects: {len(target['labels'])}")
        print(f"   Boxes shape: {target['boxes'].shape}")
        print(f"   Masks shape: {target['masks'].shape}")
        print(f"   Labels: {target['labels'].tolist()}")
        
        # Cek beberapa sample untuk statistik
        print(f"\n4. CHECKING MULTIPLE SAMPLES:")
        sample_count = min(10, len(train_dataset))
        object_counts = []
        
        for i in range(sample_count):
            try:
                _, tgt = train_dataset[i]
                object_counts.append(len(tgt['labels']))
            except Exception as e:
                print(f"   Warning: Sample {i} failed - {str(e)}")
        
        if object_counts:
            print(f"   Checked {len(object_counts)} samples")
            print(f"   Objects per tile - Min: {min(object_counts)}, Max: {max(object_counts)}, Avg: {np.mean(object_counts):.2f}")
        
        print(f"\n{'='*80}")
        print("DONE!")
        print("="*80)
        
    except Exception as e:
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
