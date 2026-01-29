# train.py
# Training loops for Center-based Single Shot Object Detection and Classification (CeSSODeC).
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
# Created:                  24-Jan-2026 15:00:00
# References:
#   None
#
# Revision history:
#   - Added TensorBoard logging (25-Jan-2026, J. Homann)
#   - Added more TB logging and confusion matrix(27-Jan-2026, J. Homann)
#
# Implemented in VSCode 1.108.1
# 2026 in the Applied Machine Learning Course Project


# TODO: IMPORTANT: Better explain what this file does/what the functions do


"""
Training loop and related functions.

Responsibilities:
- Build dataloaders
- Run training & validation loops
- Handle AMP (optional)
- Save last / best checkpoints

NO model/dataset/loss definitions here - only wiring.
"""
# QoL imports
from pathlib import Path
from typing import Any, Dict

# Torch imports
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

# Project imports
from config import RunConfig
from dataset import SingleObjectYoloDataset
from model import CeSSODeCModel
from losses import SingleObjectLoss
from checkpointIO import save_checkpoint, load_checkpoint

# For TensorBoard logging
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import visualizationHelpers as vh


# ---------------------------------------------------------
# DATALOADERS
# ---------------------------------------------------------

def build_loaders(cfg: RunConfig) -> dict[str, DataLoader]:
    """
    Build training and validation dataloaders.

    Returns:
        dict with keys:
            "train": DataLoader for training split
            "val":   DataLoader for validation split
    """

    train_dataset = SingleObjectYoloDataset(
        data_cfg=cfg.data,
        grid_cfg=cfg.grid,
        data="train",
    )

    val_dataset = SingleObjectYoloDataset(
        data_cfg=cfg.data,
        grid_cfg=cfg.grid,
        data="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return {
        "train": train_loader,
        "val": val_loader,
    }


# ---------------------------------------------------------
# TRAIN ONE EPOCH
# ---------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    cfg: RunConfig,
) -> Dict[str, float]:
    """
    Run a single training epoch. Uses automatic mixed precision if enabled in config.
    Accumulates and returns average losses over the epoch. 

    Inputs:
        model: The model to train
        loss_fn: The loss function
        optimizer: The optimizer
        loader: DataLoader for training data
        cfg: RunConfig with training settings

    Outputs:
        dict with average losses: {"Loss_center", "Loss_box", "Loss_class", "Loss_total"}
    """
    model.train()   # Model in training mode

    device = cfg.train.device  # Device from config
    amp_enabled = cfg.train.activateAMP
    scaler = GradScaler(device=device, enabled=amp_enabled)

    # Initialize loss sums and batch counter
    loss_sums = {"Loss_center": 0.0, "Loss_box": 0.0,
                 "Loss_class": 0.0, "Loss_total": 0.0}
    
    # Initialize variables for accuracy calculation (used for logging)
    n_batches = 0       # Number of batches processed
    correct = 0         # Number of correct predictions
    total = 0           # Total number of samples
    center_correct = 0  # Number of correct center predictions

    # Initilaize variables for acc tracking
    p_gt_sum = 0.0
    p_max_sum = 0.0
    p_bg_sum = 0.0
    n_samples = 0

    # Iterate over batches
    for x, gridIndices_gt, bbox_gt_norm, cls_gt in loader:
        x = x.to(device)
        gridIndices_gt = gridIndices_gt.to(device)
        bbox_gt_norm = bbox_gt_norm.to(device)
        cls_gt = cls_gt.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass with AMP
        with autocast(device_type=device, enabled=amp_enabled):
            center_pred, box_pred, cls_pred = model(x)
            losses = loss_fn(
                center_preds=center_pred,
                box_preds=box_pred,
                class_preds=cls_pred,
                gridIndices_gt=gridIndices_gt,
                bbox_gt_norm=bbox_gt_norm,
                cls_gt=cls_gt,
                gaussHm_sigma=cfg.loss.gaussHm_sigma,
                BCE_scale=cfg.loss.BCE_scale,
            )

        # Accuracy calculation (for logging)
        with torch.no_grad():
            batch_size = x.shape[0]                                                     # Current batch size

            center_flat = center_pred[:, 0].reshape(batch_size, -1)                     # Flatten center predictions
            pred_center_cell_index = torch.argmax(center_flat, dim = 1)                 # Predicted center cell index

            W = cfg.grid.W
            i_hat = pred_center_cell_index // W                                         # Predicted center cell row index
            j_hat = pred_center_cell_index % W                                          # Predicted center cell column index
            center_correct += ((i_hat == gridIndices_gt[:, 0]) & (j_hat == gridIndices_gt[:, 1])).sum().item()  # Center hit count

            cls_logits = cls_pred[torch.arange(batch_size, device=cls_pred.device), :, i_hat, j_hat]    # Class logits at predicted center
            cls_hat = torch.argmax(cls_logits, dim = 1)                                 # Predicted class at center

            correct += (cls_hat == cls_gt).sum().item()                                 # Correct class predictions
            total += batch_size                                                         # Total samples

        with torch.no_grad():
            B = x.shape[0]
            probs = torch.sigmoid(center_pred[:, 0])  # Convert logits to probabilities

            i_gt = gridIndices_gt[:, 0]
            j_gt = gridIndices_gt[:, 1]
            ar = torch.arange(B, device=device)

            p_gt = probs[ar, i_gt, j_gt]  # Probabilities at ground truth centers
            p_max = probs.view(B, -1).max(dim=1).values  # Max probabilities in each sample
            p_bg = probs.mean(dim=(1, 2))  # Mean probabilities (background)

            p_gt_sum += p_gt.sum().item()
            p_max_sum += p_max.sum().item()
            p_bg_sum += p_bg.sum().item()
            n_samples += B

    # Backward pass and optimization step
        scaler.scale(losses["Loss_total"]).backward()
        scaler.step(optimizer)
        scaler.update()
    # Loss accumulation
        for k in loss_sums:
            loss_sums[k] += losses[k].item()

        n_batches += 1

    out = {k: v / n_batches for k, v in loss_sums.items()}
    out["accuracy"] = correct / max(total, 1)
    out["center_acc"] = center_correct / max(total, 1)

    out["p_gt_avg"] = p_gt_sum / max(n_samples, 1)
    out["p_max_avg"] = p_max_sum / max(n_samples, 1)
    out["p_bg_avg"] = p_bg_sum / max(n_samples, 1)
    return out


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

@torch.no_grad()        # Disable gradient computation for validation
def validate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    loader: DataLoader,
    cfg: RunConfig,
) -> Dict[str, Any]:
    """
    Validation:
    - Compute val loss (same loss as training)
    - Decode center argmax and class argmax at that cell
    - Metrics: accuracy, center_acc, per-class accuracy
    """
    model.eval()
    device = cfg.train.device

    # Accuracy
    correct = 0
    total = 0

    # Center hit rate
    center_correct = 0

    # Per-class accuracy
    num_classes = int(getattr(cfg.model, "num_classes", 11))
    correct_per_class = torch.zeros(num_classes, dtype=torch.long)
    total_per_class = torch.zeros(num_classes, dtype=torch.long)

    # Confusion matrix counts
    confusionMatrix = torch.zeros((num_classes, num_classes), dtype=torch.long) # Here are rows: true class and columns: predicted

    # Val loss accumulation (same keys as train_one_epoch)
    loss_sums = {"Loss_center": 0.0, "Loss_box": 0.0,
                 "Loss_class": 0.0, "Loss_total": 0.0}
    total_samples = 0
    # Initialize variables for acc tracking
    p_gt_sum = 0.0
    p_max_sum = 0.0
    p_bg_sum = 0.0
    n_samples = 0

    for x, ij_gt, bbox_gt_norm, cls_gt in loader:
        x = x.to(device)
        ij_gt = ij_gt.to(device)
        bbox_gt_norm = bbox_gt_norm.to(device)
        cls_gt = cls_gt.to(device)

        center_pred, box_pred, cls_pred = model(x)

        # Average probabilities calculation
        B = x.shape[0]
        probs = torch.sigmoid(center_pred[:, 0])  # logits -> probs

        ar = torch.arange(B, device=device)
        i_gt = ij_gt[:, 0]
        j_gt = ij_gt[:, 1]

        p_gt_sum += probs[ar, i_gt, j_gt].sum().item()
        p_max_sum += probs.view(B, -1).max(dim=1).values.sum().item()
        p_bg_sum += probs.mean(dim=(1, 2)).sum().item()
        n_samples += B

        # Val losses
        losses = loss_fn(
            center_preds=center_pred,
            box_preds=box_pred,
            class_preds=cls_pred,
            gridIndices_gt=ij_gt,
            bbox_gt_norm=bbox_gt_norm,
            cls_gt=cls_gt,
            gaussHm_sigma=cfg.loss.gaussHm_sigma,
            BCE_scale=cfg.loss.BCE_scale,
        )

        # Accumulate losses 
        for k in loss_sums:
            loss_sums[k] += float(losses[k].item()) * x.shape[0]  # Sum weighted by batch size
        total_samples += x.shape[0]

        # Decode center argmax
        B = x.shape[0]
        center_flat = center_pred[:, 0].reshape(B, -1)
        idx = torch.argmax(center_flat, dim=1)

        W = cfg.grid.W
        i_hat = idx // W
        j_hat = idx % W

        # Center hit
        center_correct += ((i_hat == ij_gt[:, 0])
                           & (j_hat == ij_gt[:, 1])).sum().item()

        # Class at predicted center
        cls_logits = cls_pred[torch.arange(B, device=device), :, i_hat, j_hat]
        cls_hat = torch.argmax(cls_logits, dim=1)

        # Update confusion matrix
        for trueLabel, predictedLabel in zip(cls_gt.view(-1), cls_hat.view(-1)):    # zip() to iterate over both tensors in parallel and view(-1) to flatten to 1D vector
            tL = int(trueLabel.item())
            pL = int(predictedLabel.item())
            confusionMatrix[tL, pL] += 1               # Increment the count for the corresponding cell in the confusion matrix

        correct += (cls_hat == cls_gt).sum().item()
        total += B

        # Per-class
        for c in range(num_classes):
            mask = (cls_gt == c)
            total_per_class[c] += mask.sum().item()
            correct_per_class[c] += ((cls_hat == cls_gt) & mask).sum().item()

    acc = correct / max(total, 1)
    center_acc = center_correct / max(total, 1)

    val_losses = {k: v / max(total_samples, 1) for k, v in loss_sums.items()}

    per_class_acc = (correct_per_class.float(
    ) / torch.clamp(total_per_class.float(), min=1.0)).cpu().tolist()

    # Convert confusion matrix to numpy for easier handling outside torch
    confusionMatrixAsNumpy = confusionMatrix.cpu().numpy()


    return {
        "accuracy": acc,
        "center_acc": center_acc,
        "per_class_acc": per_class_acc,
        "confusion_matrix": confusionMatrixAsNumpy,
        **val_losses,
        "p_gt_avg": p_gt_sum / max(n_samples, 1),
        "p_max_avg": p_max_sum / max(n_samples, 1),
        "p_bg_avg": p_bg_sum / max(n_samples, 1),
    }


# ---------------------------------------------------------
# FIT LOOP
# ---------------------------------------------------------

def fit(cfg: RunConfig) -> None:
    """
    Full training loop:
    - init model / optimizer / loss
    - run epochs
    - save last & best checkpoints
    """
    # TensorBoard Writer
    writer = SummaryWriter(log_dir="runs/tensorboard") # TODO: Make dynamic with timestamp or so


    torch.manual_seed(cfg.train.seed)  # Set seed for reproducibility

    device = torch.device(cfg.train.device)  # Device from config

    loaders = build_loaders(cfg)  # Build data loaders

    # Print config summary
    print(
    f"device={cfg.train.device} | epochs={cfg.train.epochs} | batch_size={cfg.train.batch_size} | "
    f"lr={cfg.train.lr} | weight_decay={cfg.train.weight_decay} | amp={cfg.train.activateAMP} | "
    f"num_workers={cfg.train.num_workers} | imgsz={cfg.grid.imgsz} | stride_S={cfg.grid.stride_S}"
    )
    print(f"train_batches={len(loaders['train'])} | val_batches={len(loaders['val'])}")


    # Initialize model, loss function, optimizer
    model = CeSSODeCModel(cfg.model, cfg.grid).to(device)
    loss_fn = SingleObjectLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    start_epoch = 0
    best_acc = 0.0

    # Resume if last checkpoint exists
    if cfg.train.ckpt_last_path is not None and Path(cfg.train.ckpt_last_path).is_file():
        meta = load_checkpoint(cfg.train.ckpt_last_path, model, optimizer)
        start_epoch = int(meta.get("epoch", 0)) + 1
        best_acc = float(meta.get("best_acc", 0.0))

    # Train epochs and validate
    for epoch in range(start_epoch, cfg.train.epochs):
        # train one epoch
        train_metrics = train_one_epoch(
            model,
            loss_fn,
            optimizer,
            loaders["train"],
            cfg,
        )

        val_metrics = validate(
            model,
            loss_fn,
            loaders["val"],
            cfg,
        )

        print(
            f"epoch {epoch+1}/{cfg.train.epochs} | "
            f"train: total={train_metrics['Loss_total']:.4f} "
            f"center={train_metrics['Loss_center']:.4f} "
            f"box={train_metrics['Loss_box']:.4f} "
            f"class={train_metrics['Loss_class']:.4f} | "
            f"val: acc={val_metrics['accuracy']:.4f} "
            f"center_acc={val_metrics['center_acc']:.4f} "
            f"val_total={val_metrics['Loss_total']:.4f} | "
            f"best_acc={best_acc:.4f}"
        )

        # TensorBoard logging

        # Train metrics
        # Losses
        writer.add_scalar("train/Loss_total", train_metrics["Loss_total"], epoch)
        writer.add_scalar("train/Loss_center", train_metrics["Loss_center"], epoch)
        writer.add_scalar("train/Loss_box", train_metrics["Loss_box"], epoch)
        writer.add_scalar("train/Loss_class", train_metrics["Loss_class"], epoch)
        # Accuracies
        writer.add_scalar("train/accuracy", train_metrics["accuracy"], epoch)
        writer.add_scalar("train/center_acc", train_metrics["center_acc"], epoch)

        # Average probabilities
        writer.add_scalar("train/p_gt_avg", train_metrics["p_gt_avg"], epoch)
        writer.add_scalar("train/p_max_avg", train_metrics["p_max_avg"], epoch)
        writer.add_scalar("train/p_bg_avg", train_metrics["p_bg_avg"], epoch)
        
        # Val metrics
        # Losses
        writer.add_scalar("val/Loss_total", val_metrics["Loss_total"], epoch)
        writer.add_scalar("val/Loss_center", val_metrics["Loss_center"], epoch)
        writer.add_scalar("val/Loss_box", val_metrics["Loss_box"], epoch)
        writer.add_scalar("val/Loss_class", val_metrics["Loss_class"], epoch)
        # Accuracies
        writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
        writer.add_scalar("val/center_acc", val_metrics["center_acc"], epoch)
        writer.flush() # Ensure data is written to disk

        #Average probabilities on val set
        writer.add_scalar("val/p_gt_avg", val_metrics["p_gt_avg"], epoch)
        writer.add_scalar("val/p_max_avg", val_metrics["p_max_avg"], epoch)
        writer.add_scalar("val/p_bg_avg", val_metrics["p_bg_avg"], epoch)

        # Confusion Matrix
        class_names = getattr(cfg.model, "class_names", None) # List of class names, if empty read from config
        if class_names is None:
            class_names = [f"class_{i}" for i in range(int(getattr(cfg.model, "num_classes", 11)))]

        figure_confusionMatrix = vh.plotConfMatrix(val_metrics["confusion_matrix"], class_names)   # Plot confusion matrix via helper function
        writer.add_figure("val/confusion_matrix", figure_confusionMatrix, epoch)                # Log confusion matrix figure to TensorBoard
        plt.close(figure_confusionMatrix)   # Close the figure to free memory
        writer.flush()                      # Ensure data is written to disk

        # Predictions vs Ground Truth Visualization
        if epoch % 1 == 0:  # Log every n epochs to save space
            pred_vs_gt_visualization = vh.visualize_pred_vs_gt(model, loaders["val"], cfg, images2visualize=4)  # Visualize predictions vs ground truth via helper function
            writer.add_image("val/pred_vs_gt", pred_vs_gt_visualization, epoch)  # Log the image grid to TensorBoard
            writer.flush()  # Ensure data is written to disk


        acc = val_metrics["accuracy"]

        # save last
        save_checkpoint(
            cfg.train.ckpt_last_path,
            model,
            optimizer,
            meta={
                "epoch": epoch,
                "best_acc": best_acc,
                "train": train_metrics,
                "val": val_metrics,
            },
        )

        # Save best checkpoint
        if acc > best_acc and cfg.train.ckpt_best_path is not None:
            best_acc = acc
            print(f"  new best: epoch={epoch+1} | best_acc={best_acc:.4f} -> saving best checkpoint")
            save_checkpoint(
                cfg.train.ckpt_best_path,
                model,
                optimizer,
                meta={
                    "epoch": epoch,
                    "best_acc": best_acc,
                },
            )
    # Close TensorBoard writer
    writer.close()

    return



