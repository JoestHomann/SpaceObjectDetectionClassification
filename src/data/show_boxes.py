import os
import random
import cv2
import matplotlib.pyplot as plt

# Pfade relativ zum Skript
IMAGES_DIR = "images"
LABELS_DIR = "labels"
NUM_SAMPLES = 10

# Unterordner automatisch erkennen (z. B. train/)
def find_image_files(images_dir):
    image_files = []
    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(root, f))
    return image_files

image_files = find_image_files(IMAGES_DIR)
samples = random.sample(image_files, min(NUM_SAMPLES, len(image_files)))

for img_path in samples:
    # Bild laden
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # passenden Label-Pfad bauen
    rel_path = os.path.relpath(img_path, IMAGES_DIR)
    label_path = os.path.join(LABELS_DIR, os.path.splitext(rel_path)[0] + ".txt")

    # Bounding Boxes zeichnen
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                cls, xc, yc, bw, bh = map(float, line.split())

                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    img,
                    f"class {int(cls)}",
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )

    # Anzeigen
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.title(os.path.basename(img_path))
    plt.show()
