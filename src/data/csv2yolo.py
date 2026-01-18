from pathlib import Path
import csv
import ast
from PIL import Image

# File paths as variables
dataset_path = Path("D:/SpaceObjectDetection-YOLO/data/spark-2022-stream-1")      # Spark 2022 Stream 1 Dataset Directory
labels_path = dataset_path / "labels"                            # Labels Directory

train_csv_path = labels_path / "train.csv"                       # Training Dataset Labels CSV File
val_csv_path = labels_path / "val.csv"                           # Validation Dataset Labels CSV File

train_img_path = dataset_path / "train" / "train"                # Training Images Directory
val_img_path = dataset_path / "val" / "val"                      # Validation Images Directory

output_path = dataset_path / "labels_yolo"                       # Output Directory For YOLO Formatted Labels
output_train_path = output_path / "train_yolo"                   # Output Directory For Training YOLO Labels
output_val_path = output_path / "val_yolo"                       # Output Directory For Validation YOLO Labels



# Open the CSV file and read its contents into a list of dictionaries
def read_csv_rows(filepath):
    with filepath.open('r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        csv_list = list(reader)
    return csv_list

# Convert bbox from [x1, y1, x2, y2] to float values
def parse_bbox_xyxy(bbox_str):
    # bbox_str example: "[183, 311, 657, 415]"
    x1, y1, x2, y2 = ast.literal_eval(bbox_str)
    return float(x1), float(y1), float(x2), float(y2)

# Get image size (width, height) from image file
def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size  # (width, height)

# Convert bbox from [x1, y1, x2, y2] to YOLO normalized format [xc, yc, bw, bh]
def xyxy_to_yolo_norm(x1, y1, x2, y2, w, h):
    bw = x2 - x1
    bh = y2 - y1
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return xc / w, yc / h, bw / w, bh / h

# Build class map from training and validation rows
def build_class_map(train_rows, val_rows):
    classes = sorted({r["class"] for r in (train_rows + val_rows) if r["class"]})
    return {c: i for i, c in enumerate(classes)}

# Write class names to file
def write_class_names(out_root, class_map):
    names = [c for c, _ in sorted(class_map.items(), key=lambda kv: kv[1])]
    (out_root / "class_names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


def csv_filename_to_jpg_path(img_dir, csv_filename):
    stem = Path(csv_filename).stem
    return img_dir / f"{stem}.jpg"

def write_yolo_labels(rows, img_dir, out_dir, class_map):
    # Group annotations by filename
    by_file = {}
    for r in rows:
        fn = r["filename"]
        by_file.setdefault(fn, []).append(r)

    missing_images = 0
    written_files = 0
    written_boxes = 0

    total_files = len(by_file)
    for i, (fn, ann_list) in enumerate(by_file.items(), start=1):
        # Progress print every 1000 files
        if i % 1000 == 0 or i == 1 or i == total_files:
            print(f"[{i}/{total_files}] processed={written_files} missing={missing_images} boxes={written_boxes}")

        img_path = csv_filename_to_jpg_path(img_dir, fn)
        if not img_path.exists():
            missing_images += 1
            continue

        w, h = get_image_size(img_path)
        lines = []

        for ann in ann_list:
            cls = ann["class"]
            bbox_str = ann["bbox"]

            x1, y1, x2, y2 = parse_bbox_xyxy(bbox_str)
            xc, yc, bw, bh = xyxy_to_yolo_norm(x1, y1, x2, y2, w, h)

            # Skip degenerate boxes
            if bw <= 0.0 or bh <= 0.0:
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



def main():

    # Resolve Dataset Path
    print("Dataset Path:", dataset_path.resolve())

    # Verify Existence Of Paths
    print("Training CSV Path:", train_csv_path, "Exists:", train_csv_path.exists())
    print("Validation CSV Path:", val_csv_path, "Exists:", val_csv_path.exists())
    print("Training Images Path:", train_img_path, "Exists:", train_img_path.exists())
    print("Validation Images Path:", val_img_path, "Exists:", val_img_path.exists())

    # Resolve Output Path
    print("Output Path:", output_path.resolve())

    # Verify Existence Of Output Paths
    print("Output Training Path:", output_train_path, "Exists:", output_train_path.exists())
    print("Output Validation Path:", output_val_path, "Exists:", output_val_path.exists())


    # Load CSV files
    train_rows = read_csv_rows(train_csv_path)
    val_rows   = read_csv_rows(val_csv_path)
    
    # Inspect columns
    print("Train CSV columns:", list(train_rows[0].keys()) if train_rows else "EMPTY")
    print("Val CSV columns:  ", list(val_rows[0].keys()) if val_rows else "EMPTY")

    # Preview a first row of each CSV
    print("Train first row:", train_rows[0] if train_rows else "EMPTY")
    print("Val first row:  ", val_rows[0] if val_rows else "EMPTY")

    # Build class map and write class names file
    class_map = build_class_map(train_rows, val_rows)
    write_class_names(output_path, class_map)
    print("Number of classes:", len(class_map))

    # Convert and write YOLO formatted labels
    print("Converting train -> YOLO...")
    write_yolo_labels(train_rows, train_img_path, output_train_path, class_map)
    print("Converting val -> YOLO...")
    write_yolo_labels(val_rows, val_img_path, output_val_path, class_map)

    print("Done.")



if __name__ == "__main__":
    main()

