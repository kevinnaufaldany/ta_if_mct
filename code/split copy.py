import os
import json
import cv2
import numpy as np
from shapely.geometry import Polygon, box

INPUT_DIR = "testset"
OUTPUT_DIR = "testset_split"
GRID = (2, 2)  # 2x2 = 4 tile

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clip_polygon(points, crop_box):
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
        print(f"Warning: Skipping invalid polygon - {e}")
        return None

for file in os.listdir(INPUT_DIR):
    if not file.endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(INPUT_DIR, file)
    json_path = os.path.join(INPUT_DIR, os.path.splitext(file)[0] + ".json")

    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    with open(json_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    tile_h = h // GRID[0]
    tile_w = w // GRID[1]

    tile_id = 0
    for i in range(GRID[0]):
        for j in range(GRID[1]):
            x1 = j * tile_w
            y1 = i * tile_h
            x2 = x1 + tile_w
            y2 = y1 + tile_h

            crop = image[y1:y2, x1:x2]
            crop_box = box(x1, y1, x2, y2)

            new_shapes = []
            for shape in ann["shapes"]:
                if shape["shape_type"] != "polygon":
                    continue

                clipped = clip_polygon(shape["points"], crop_box)
                if clipped is None:
                    continue

                # geser koordinat ke lokal tile
                clipped[:, 0] -= x1
                clipped[:, 1] -= y1

                new_shapes.append({
                    "label": shape["label"],
                    "points": clipped.tolist(),
                    "shape_type": "polygon"
                })

            if len(new_shapes) == 0:
                continue

            out_img = f"{os.path.splitext(file)[0]}_tile{tile_id}.png"
            out_json = f"{os.path.splitext(file)[0]}_tile{tile_id}.json"

            cv2.imwrite(os.path.join(OUTPUT_DIR, out_img), crop)

            out_ann = {
                "version": ann.get("version", "unknown"),
                "shapes": new_shapes
            }

            with open(os.path.join(OUTPUT_DIR, out_json), "w", encoding="utf-8") as f:
                json.dump(out_ann, f, indent=2)

            tile_id += 1

print("✅ Dataset berhasil di-split")
