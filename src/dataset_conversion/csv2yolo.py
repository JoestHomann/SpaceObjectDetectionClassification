from pathlib import Path
import csv
import ast
from PIL import Image
import shutil

# File paths as variables
dataset_path = Path("D:/SpaceObjectDetection-YOLO/data/spark-2022-stream-1")    # Spark 2022 Stream 1 Dataset Directory
labels_path = dataset_path / "labels"                                           # Labels Directory

train_csv_path = labels_path / "train.csv"                                      # Training Dataset Labels CSV File
val_csv_path   = labels_path / "val.csv"                                        # Validation Dataset Labels CSV File

train_img_path = dataset_path / "train" / "train"                               # Training Images Directory
val_img_path   = dataset_path / "val" / "val"                                   # Validation Images Directory

output_path       = dataset_path / "labels"                                     # Output Directory For YOLO Formatted Labels
output_train_path = output_path / "train"                                       # Output Directory For Training YOLO Labels
output_val_path   = output_path / "val"                                         # Output Directory For Validation YOLO Labels


# Helper functions for CSV reading and bbox conversion
def read_csv_rows(filepath: Path):
    with filepath.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)

# Spark 2022 Stream 1 bbox appears to be stored as [y1, x1, y2, x2] (yxyx), not [x1, y1, x2, y2].
def parse_bbox_spark_yxyx_to_xyxy(bbox_str: str):
    # bbox_str example: "[183, 311, 657, 415]"
    y1, x1, y2, x2 = ast.literal_eval(bbox_str) #converts string representation of list to actual list
    return float(x1), float(y1), float(x2), float(y2)

def get_image_size(image_path: Path):
    with Image.open(image_path) as img:
        return img.size  # (width, height)

def clamp(v, lo, hi):                               #clamps value to be within lo and hi (in this case image bounds)
    return lo if v < lo else hi if v > hi else v

def sanitize_xyxy(x1, y1, x2, y2, w, h):
    # Ensure correct ordering
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    # Clamp to image bounds
    x1 = clamp(x1, 0.0, w - 1.0)
    x2 = clamp(x2, 0.0, w - 1.0)
    y1 = clamp(y1, 0.0, h - 1.0)
    y2 = clamp(y2, 0.0, h - 1.0)

    return x1, y1, x2, y2

def xyxy_to_yolo_norm(x1, y1, x2, y2, w, h):    #Convert xyxy to YOLO format (xc, yc, bw, bh) normalized to [0,1]
    bw = x2 - x1
    bh = y2 - y1
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return xc / w, yc / h, bw / w, bh / h

def build_class_map(train_rows, val_rows):   # Build class name to ID mapping from all classes in train and val sets
    classes = sorted({r["class"] for r in (train_rows + val_rows) if r.get("class")})
    return {c: i for i, c in enumerate(classes)}

def write_class_names(out_root: Path, class_map):   # Write class names to class_names.txt in output directory
    names = [c for c, _ in sorted(class_map.items(), key=lambda kv: kv[1])]
    (out_root / "class_names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

def csv_filename_to_jpg_path(img_dir: Path, csv_filename: str):
    # In Spark CSV, filename might be "img0123.png" or "img0123.jpg" etc.
    # Your dataset images are .jpg, so we map by stem -> stem.jpg
    stem = Path(csv_filename).stem
    return img_dir / f"{stem}.jpg"

def write_yolo_labels(rows, img_dir: Path, out_dir: Path, class_map):   # Write YOLO formatted label files from CSV rows
    # Group annotations by filename
    by_file = {}
    for r in rows:
        fn = r.get("filename")
        if not fn:
            continue
        by_file.setdefault(fn, []).append(r)

    # Tracking stats
    missing_images = 0
    written_files = 0
    written_boxes = 0

    total_files = len(by_file)
    for i, (fn, ann_list) in enumerate(by_file.items(), start=1):
        if i % 1000 == 0 or i == 1 or i == total_files:     # Progress print every 1000 files, and first/last
            print(f"[{i}/{total_files}] processed={written_files} missing={missing_images} boxes={written_boxes}")

        img_path = csv_filename_to_jpg_path(img_dir, fn)    #missing images handled here
        if not img_path.exists():
            missing_images += 1
            continue

        w, h = get_image_size(img_path)
        lines = []

        for ann in ann_list:
            cls = ann.get("class")
            bbox_str = ann.get("bbox")

            if not cls or not bbox_str:
                continue

            # Spark bbox: [y1, x1, y2, x2] -> convert to xyxy
            x1, y1, x2, y2 = parse_bbox_spark_yxyx_to_xyxy(bbox_str)
            x1, y1, x2, y2 = sanitize_xyxy(x1, y1, x2, y2, w, h)

            xc, yc, bw, bh = xyxy_to_yolo_norm(x1, y1, x2, y2, w, h)

            # Skip degenerate boxes
            if bw <= 0.0 or bh <= 0.0:
                continue

            # Skip boxes that end up outside [0,1] due to bad input (after clamp they should be OK,
            # but bw/bh could still be near-zero)
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= bw <= 1.0 and 0.0 <= bh <= 1.0):
                continue

            class_id = class_map[cls]
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        label_path = out_dir / (Path(fn).stem + ".txt")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        written_files += 1
        written_boxes += len(lines)

    print("Missing images:", missing_images)
    print("Label files written:", written_files)
    print("Total boxes written:", written_boxes)

def ensure_yolo_layout_and_copy_files(dataset_root: Path):  # Ensure YOLO directory layout and copy image files to correct locations
    # Create YOLO layout directories  
    images_train = dataset_root / "images" / "train"
    images_val   = dataset_root / "images" / "val"
    labels_train = dataset_root / "labels" / "train"
    labels_val   = dataset_root / "labels" / "val"
    # Make directories if they don't exist
    images_train.mkdir(parents=True, exist_ok=True)
    images_val.mkdir(parents=True, exist_ok=True)
    labels_train.mkdir(parents=True, exist_ok=True)
    labels_val.mkdir(parents=True, exist_ok=True)
    # Source image directories
    src_images_train = dataset_root / "train" / "train"
    src_images_val   = dataset_root / "val" / "val"
    # Copy image files to YOLO layout directories
    def copy_all_files(src_dir: Path, dst_dir: Path, pattern="*"):
        if not src_dir.exists():
            print("Missing source:", src_dir)
            return 0
        count = 0
        for p in src_dir.glob(pattern):
            if p.is_file():
                dst = dst_dir / p.name
                if not dst.exists():
                    shutil.copy2(p, dst)
                    count += 1
        return count
    # Copy train and val images
    n_train_imgs = copy_all_files(src_images_train, images_train, "*.jpg")
    n_val_imgs   = copy_all_files(src_images_val, images_val, "*.jpg")
    # Copy class_names.txt to configs/ directory
    class_names_src = dataset_root / "labels" / "class_names.txt"
    class_names_dst = Path("configs") / "class_names.txt"
    if class_names_src.exists():
        class_names_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(class_names_src, class_names_dst)
    # Print summary
    print("YOLO layout copied:")
    print(" train images copied:", n_train_imgs)
    print(" val images copied:  ", n_val_imgs)
    print(" images/train dir:", images_train)
    print(" labels/train dir:", labels_train)

def main(): # Main conversion function
    print("Dataset Path:", dataset_path.resolve())
    # Check existence of input paths
    print("Training CSV Path:", train_csv_path, "Exists:", train_csv_path.exists())
    print("Validation CSV Path:", val_csv_path, "Exists:", val_csv_path.exists())
    print("Training Images Path:", train_img_path, "Exists:", train_img_path.exists())
    print("Validation Images Path:", val_img_path, "Exists:", val_img_path.exists())
    # Output paths
    print("Output Path:", output_path.resolve())
    print("Output Training Path:", output_train_path, "Exists:", output_train_path.exists())
    print("Output Validation Path:", output_val_path, "Exists:", output_val_path.exists())

    train_rows = read_csv_rows(train_csv_path)
    val_rows   = read_csv_rows(val_csv_path)
    # Debug prints
    print("Train CSV columns:", list(train_rows[0].keys()) if train_rows else "EMPTY")
    print("Val CSV columns:  ", list(val_rows[0].keys()) if val_rows else "EMPTY")
    print("Train first row:", train_rows[0] if train_rows else "EMPTY")
    print("Val first row:  ", val_rows[0] if val_rows else "EMPTY")

    output_path.mkdir(parents=True, exist_ok=True)
    output_train_path.mkdir(parents=True, exist_ok=True)
    output_val_path.mkdir(parents=True, exist_ok=True)

    class_map = build_class_map(train_rows, val_rows)
    write_class_names(output_path, class_map)
    print("Number of classes:", len(class_map))
    
    print("Converting train -> YOLO...")
    write_yolo_labels(train_rows, train_img_path, output_train_path, class_map)
    print("Converting val -> YOLO...")
    write_yolo_labels(val_rows, val_img_path, output_val_path, class_map)

    ensure_yolo_layout_and_copy_files(dataset_path)

    print("Done.")

if __name__ == "__main__":
    main()
