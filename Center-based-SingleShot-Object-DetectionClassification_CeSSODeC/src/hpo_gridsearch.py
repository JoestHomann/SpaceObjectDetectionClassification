# hpo_gridsearch.py
# Hyperparameter optimization script using grid search for Center-based Single Shot Object Detection and Classification (CeSSODeC)
# 
# Details:
#   None
# 
# Syntax:  
# 
# Inputs:
#   None
#
# Outputs:
#   None
#
# Examples:
#   None
#
# See also:
#   None
#
# Author:                   J. Homann, C. Kern, F. Kolb
# Email:                    st171800@stud.uni-stuttgart.de
# Created:                  28-Jan-2026 18:20:00
# References:
#   None
#
# Revision history:
#   None
#
# Implemented in VSCode 1.108.1
# 2026 in the Applied Machine Learning Course Project


import itertools
import json
import subprocess
from pathlib import Path
import torch

# -------------------------------
# Grid definitions for HPO
# -------------------------------

LR_GRID = [1e-4, 3e-4]          # Learning rate
WD_GRID = [1e-5, 1e-4]          # Weight decay
SIGMA_GRID = [1.0, 2.0, 3.0]    # Gaussian heatmap standard deviation
K_GRID = [10.0, 25.0, 40.0]      # Positive sample weight for BCE loss

EPOCHS = 15   # short runs for HPO
DATASET_ROOT = "D:/SpaceObjectDetection-YOLO/data/spark-2022-stream-1"

BASE_RUN_DIR = Path("D:/SpaceObjectDetection-YOLO/Center-based-SingleShot-Object-DetectionClassification_CeSSODeC/runs/hpo_grid")
BASE_RUN_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------
# Grid search loop
# -------------------------------

hpo_id = 0
hpo_results = []

for lr, weight_decay, sigma, k in itertools.product(
        LR_GRID, WD_GRID, SIGMA_GRID, K_GRID):

    # Create unique run directory for each HPO run
    hpo_run_name = f"hpo_{hpo_id:04d}"
    hpo_run_dir = BASE_RUN_DIR / hpo_run_name
    hpo_run_dir.mkdir(parents=True, exist_ok=True)
    
    # Define checkpoint paths
    last_checkpoint = hpo_run_dir / "last.pth"
    best_checkpoint = hpo_run_dir / "best.pth"
    
    # Build command to run training with current hyperparameters
    command = [
        "python", "train_main.py",
        "--datasetRoot", DATASET_ROOT,
        "--epochs", str(EPOCHS),
        "--lr", str(lr),
        "--weight_decay", str(weight_decay),
        "--gaussHm_sigma", str(sigma),
        "--BCE_scale", str(k),
        "--checkpointPath_last", str(last_checkpoint),
        "--checkpointPath_best", str(best_checkpoint),
    ]
    # Print HPO run details and execute the training command as a subprocess
    print(f"[{hpo_run_name}] lr={lr} weight_decay={weight_decay} sigma={sigma} k={k}")
    subprocess.run(command, check=True)

    # Read best_acc from checkpoint meta
    checkpoint = torch.load(best_checkpoint, map_location="cpu")
    meta = checkpoint.get("meta", {})
    best_acc = float(meta.get("best_acc", 0.0))

    # Store HPO results in list
    hpo_results.append({
        "trial": hpo_run_name,
        "lr": lr,
        "weight_decay": weight_decay,
        "gaussHm_sigma": sigma,
        "BCE_scale": k,
        "best_acc": best_acc,
    })

    # Save parameters and scores to JSON file in run directory
    with open(hpo_run_dir / "parameters_and_scores.json", "w") as f:
        json.dump(hpo_results[-1], f, indent=2)
    
    # Increment HPO ID for next run
    hpo_id += 1

# -------------------------------
# Summary
# -------------------------------

# Sort results by best accuracy in descending order
hpo_results = sorted(hpo_results, key=lambda x: x["best_acc"], reverse=True)

# Save summary of top HPO results to JSON file
with open(BASE_RUN_DIR / "summary.json", "w") as f:
    json.dump(hpo_results, f, indent=2)
print("\nTOP 5 CONFIGS:")
for r in hpo_results[:5]:
    print(r)


