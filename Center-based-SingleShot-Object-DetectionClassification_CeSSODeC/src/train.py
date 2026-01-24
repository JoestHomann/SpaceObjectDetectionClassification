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
#   None
#
# Implemented in VSCode 1.108.1
# 2026 in the Applied Machine Learning Course Project



# Was macht die train.py?
# Ganz einfach, sie verbindet Dataloader, Model und Loss Function miteinander.

"""
Training orchestration for center_singleobj_v1_modular.

Responsibilities:
- Build dataloaders
- Run training & validation loops
- Handle AMP (optional)
- Save last / best checkpoints

NO model/dataset/loss definitions here - only wiring.
"""

from typing import Dict
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from config import RunConfig
from dataset import SingleObjectYoloDataset
from model import CeSSODeCModel
from losses import SingleObjectLoss
from checkpointIO import save_checkpoint, load_checkpoint

# ---------------------------------------------------------
# DATALOADERS
# ---------------------------------------------------------

from torch.utils.data import DataLoader
from dataset import SingleObjectYoloDataset
from config import RunConfig

# Was macht der Dataloader?
# Er lädt die Daten in Batches und bereitet sie für das Training vor.
def build_loaders(cfg: RunConfig) -> dict[str, DataLoader]:
    """
    Builds PyTorch DataLoaders for training and validation.

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

    device = cfg.train.device #Device from config
    amp_enabled = cfg.train.activateAMP 
    scaler = GradScaler(enabled=amp_enabled)

    # Initialize loss sums and batch counter
    loss_sums = {"Loss_center": 0.0, "Loss_box": 0.0, "Loss_class": 0.0, "Loss_total": 0.0} 
    n_batches = 0

    # Iterate over batches
    for x, gridIndices_gt, bbox_gt_norm, cls_gt in loader:
        x = x.to(device)
        gridIndices_gt = gridIndices_gt.to(device)
        bbox_gt_norm = bbox_gt_norm.to(device)
        cls_gt = cls_gt.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        with autocast(enabled=amp_enabled):
            center_pred, box_pred, cls_pred = model(x)
            losses = loss_fn(
                center_pred=center_pred,
                box_pred=box_pred,
                cls_pred=cls_pred,
                gridIndices_gt=gridIndices_gt,
                bbox_gt_norm=bbox_gt_norm,
                cls_gt=cls_gt,
            )
    # Backward pass and optimization step
        scaler.scale(losses["L"]).backward()
        scaler.step(optimizer)
        scaler.update()
    # Loss accumulation
        for k in loss_sums:
            loss_sums[k] += losses[k].item()

        n_batches += 1

    return {k: v / n_batches for k, v in loss_sums.items()}

# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: RunConfig,
) -> Dict[str, float]:
    """
    Validation using v1 decode:
    - center: argmax over grid
    - class: argmax at that cell
    Metric:
    - classification accuracy
    """
    model.eval() # Model in evaluation mode
    device = cfg.train.device

    correct = 0
    total = 0

    # Iterate over validation batches
    for x, ij_gt, _, cls_gt in loader:
        x = x.to(device)
        ij_gt = ij_gt.to(device)
        cls_gt = cls_gt.to(device)

        center_pred, _, cls_pred = model(x)

        B = x.shape[0]

        # flatten center heatmap
        center_flat = center_pred[:, 0].reshape(B, -1)
        idx = torch.argmax(center_flat, dim=1)

        H, W = cfg.grid.H, cfg.grid.W
        i_hat = idx // W
        j_hat = idx % W

        cls_logits = cls_pred[torch.arange(B), :, i_hat, j_hat]
        cls_hat = torch.argmax(cls_logits, dim=1)

        correct += (cls_hat == cls_gt).sum().item()
        total += B

    acc = correct / max(total, 1)
    return {"accuracy": acc}


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
    torch.manual_seed(cfg.train.seed) # Set seed for reproducibility
 
    device = torch.device(cfg.train.device) # Device from config

    loaders = build_loaders(cfg) # Build data loaders

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
    if cfg.train.ckpt_last_path is not None:
        meta = load_checkpoint(
            cfg.train.ckpt_last_path,
            model,
            optimizer,
        )
        if meta is not None:
            start_epoch = meta.get("epoch", 0) + 1
            best_acc = meta.get("best_acc", 0.0)
    
    # Train epochs and validate
    for epoch in range(start_epoch, cfg.train.epochs):
        #train one epoch
        train_metrics = train_one_epoch(
            model,
            loss_fn,
            optimizer,
            loaders["train"],
            cfg,
        )

        val_metrics = validate(
            model,
            loaders["val"],
            cfg,
        )

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

        # save best
        if acc > best_acc and cfg.train.ckpt_best_path is not None:
            best_acc = acc
            save_checkpoint(
                cfg.train.ckpt_best_path,
                model,
                optimizer,
                meta={
                    "epoch": epoch,
                    "best_acc": best_acc,
                },
            )





# def build_model_loss_optim(cfg: RunConfig) -> Tuple[CenterSingleObjNet, CenterSingleObjLoss, torch.optim.Optimizer]:
#     """
#     Build model, loss function, and optimizer.

#     Inputs:
#       cfg: RunConfig

#     Outputs:
#       (model, loss_fn, optimizer)
#     """
#     device = torch.device(cfg.train.device)

#     model = CeSSODeCModel(cfg.model, cfg.grid).to(device)
#     loss_fn = SingleObjectLoss(eps=1e-6, box_weight=5.0).to(device)

#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=cfg.train.lr,
#         weight_decay=cfg.train.weight_decay,
#     )

#     return model, loss_fn, optimizer TODO: Braucht man die Funktion evtl?



# def _set_seed(seed: int) -> None:
#     # Make runs more reproducible.
#     random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed) TODO: Funktion kann implementiert werden, um Seed besser zu setzen

