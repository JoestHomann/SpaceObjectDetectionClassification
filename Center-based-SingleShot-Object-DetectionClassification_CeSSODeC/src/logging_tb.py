# logging_tb.py
# Outsourced logging functions for tensorboard from train.py for better code organization
# 
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
# Author:                   J. Homann
# Email:                    st171800@stud.uni-stuttgart.de
# Created:                  30-Jan-2026 15:00:00
# References:
#   None
#
# Revision history:
#   None
#
# Implemented in VSCode 1.108.1
# 2026 in the Applied Machine Learning Course Project

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import visualizationHelpers as vh


def tb_writer_setup(logs_dir: Path) -> SummaryWriter:
    """Sets up TensorBoard writer with a timestamped log directory.

    Inputs:
        logs_dir (Path): Base directory for logs

    Outputs:
        SummaryWriter: Configured TensorBoard SummaryWriter
    """

    # Create a timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create log directory with timestamp
    tb_log_dir = logs_dir / f"run_{timestamp}"
    tb_log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=tb_log_dir)

    return writer


# Log training metrics
def log_train_metrics(writer: SummaryWriter, epoch: int, train_metrics: dict[str, Any]) -> None:
    """
    Logs training metrics to TensorBoard.

    Inputs:
        writer (SummaryWriter): TensorBoard SummaryWriter
        epoch (int): Current epoch number
        train_metrics (dict): Dictionary containing training metrics

    Outputs:
        None
    """
    # Log scalar training metrics
    writer.add_scalar("train/Loss_total", train_metrics["Loss_total"], epoch)
    writer.add_scalar("train/Loss_center", train_metrics["Loss_center"], epoch)
    writer.add_scalar("train/Loss_box", train_metrics["Loss_box"], epoch)
    writer.add_scalar("train/Loss_class", train_metrics["Loss_class"], epoch)
    writer.add_scalar("train/accuracy", train_metrics["accuracy"], epoch)
    writer.add_scalar("train/center_acc", train_metrics["center_acc"], epoch)

    writer.add_scalar("train/center_prob_at_gt_avg", train_metrics["center_prob_at_gt_avg"], epoch)
    writer.add_scalar("train/center_prob_max_avg", train_metrics["center_prob_max_avg"], epoch)
    writer.add_scalar("train/center_prob_background_avg", train_metrics["center_prob_background_avg"], epoch)

    writer.flush()  # Ensure data is written to disk

    return None


# Log validation metrics
def log_val_metrics(writer: SummaryWriter, epoch: int, val_metrics: dict[str, Any]) -> None:
    """
    Logs validation metrics to TensorBoard.

    Inputs:
        writer (SummaryWriter): TensorBoard SummaryWriter
        epoch (int): Current epoch number
        val_metrics (dict): Dictionary containing validation metrics

    Outputs:
        None
    """
    # Log scalar validation metrics
    writer.add_scalar("val/Loss_total", val_metrics["Loss_total"], epoch)
    writer.add_scalar("val/Loss_center", val_metrics["Loss_center"], epoch)
    writer.add_scalar("val/Loss_box", val_metrics["Loss_box"], epoch)
    writer.add_scalar("val/Loss_class", val_metrics["Loss_class"], epoch)
    writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
    writer.add_scalar("val/center_acc", val_metrics["center_acc"], epoch)

    writer.add_scalar("val/center_prob_at_gt_avg", val_metrics["center_prob_at_gt_avg"], epoch)
    writer.add_scalar("val/center_prob_max_avg", val_metrics["center_prob_max_avg"], epoch)
    writer.add_scalar("val/center_prob_background_avg", val_metrics["center_prob_background_avg"], epoch)

    writer.flush()  # Ensure data is written to disk
        
    return None


# Log confusion matrix
def log_confusion_matrix(writer: SummaryWriter, epoch: int, confusion_matrix, class_names: list[str], tag: str = "val/confusion_matrix") -> None:
    """
    Logs confusion matrix to TensorBoard.

    Inputs:
        writer (SummaryWriter): TensorBoard SummaryWriter
        epoch (int): Current epoch number
        confusion_matrix (np.ndarray): Confusion matrix data
        class_names (list[str]): List of class names
        tag (str): Tag for the confusion matrix in TensorBoard
    Outputs:
        None
    """
    # Create and log confusion matrix figure
    figure_confusingMatrix = vh.plotConfMatrix(confusion_matrix, class_names)
    writer.add_figure(tag, figure_confusingMatrix, epoch)     

    plt.close(figure_confusingMatrix)  # Close the figure to free memory

    writer.flush()  # Ensure data is written to disk

    return None


# Log prediction vs ground truth visualization
def log_pred_vs_gt(writer: SummaryWriter, epoch: int, model, loader_val, cfg, class_names: list[str], images2visualize: int = 16, tag: str = "val/pred_vs_gt") -> None:
    """
    Logs prediction vs ground truth visualization to TensorBoard.

    Inputs:
        writer (SummaryWriter): TensorBoard SummaryWriter
        epoch (int): Current epoch number
        model: Trained model
        loader_val: Validation data loader
        cfg: Configuration object
        class_names (list[str]): List of class names
        images2visualize (int): Number of images to visualize
        tag (str): Tag for the visualization in TensorBoard

    Outputs:
        None    
    """

    grid = vh.visualize_pred_vs_gt(
        model,
        loader_val,
        cfg,
        images2visualize=images2visualize,
        class_names=class_names,
    )
    writer.add_image(tag, grid, epoch)

    writer.flush()  # Ensure data is written to disk

    return None

