# visualizationHelpers.py
#   Visualization helpers for Center-based Single Shot Object Detection and Classification (CeSSODeC).
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

from cmath import rect
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from config import RunConfig

from PIL import Image
import matplotlib.patches as patches



# def plotHeatmap # TODO: Implement heatmap plotting function
# def plotBoundingBoxes # TODO: Implement bounding box plotting function


# Plotting function for confusion matrix
def plotConfMatrix(confusion_matrix: np.ndarray, class_names: list[str], normalize: bool = True):
    """
    Plots a confusion matrix using Matplotlib.

    Args:
        confusion_matrix (np.ndarray): The confusion matrix to plot.
        class_names (list[str]): List of class names corresponding to the matrix indices.

    Returns:
        plt.Figure: The Matplotlib figure containing the confusion matrix plot.
    """
    confusionMatrix = confusion_matrix.astype(np.float64)

    # Normalize the confusion matrix if specified
    if normalize:
        row_sums = confusionMatrix.sum(axis=1, keepdims=True)               # Sum of each row (true labels)
        confusionMatrix = confusionMatrix / np.clip(row_sums, 1.0, None)    # Normalize each row to sum to 1, avoid division by zero by clipping


    figure, axes = plt.subplots(figsize=(8, 8))                 # Set figure size and axes
    imageObject = axes.imshow(confusionMatrix, interpolation='nearest')  # Display the confusion matrix as an image with nearest neighbor interpolation

    axes.set_title("Confusion Matrix"+(" (normalized)" if normalize else ""))   # Set title of the plot
    figure.colorbar(imageObject, ax=axes, fraction=0.046, pad=0.04)  # Add colorbar to the plot

    tick_marks = np.arange(len(class_names))                     # Create tick marks for each class
    axes.set_xticks(tick_marks)                                  # Set x-ticks
    axes.set_yticks(tick_marks)                                  # Set y-ticks
    axes.set_xticklabels(class_names, rotation=45, ha="right")   # Set x-tick labels with rotation
    axes.set_yticklabels(class_names)                            # Set y-tick labels

    axes.set_ylabel("True label")                                # Set y-axis label
    axes.set_xlabel("Predicted label")                           # Set x-axis label

    figure.tight_layout()                                        # Adjust layout to prevent overlap

    return figure

def _denormalize_for_tb(x: torch.Tensor, config: RunConfig) -> torch.Tensor:
    """
    Denormalizes an image tensor for visualization in TensorBoard.

    Inputs:
        - x: Normalized image tensor (B, C, H, W)
        - config: RunConfig object containing data normalization settings

    Outputs:
        - x: Denormalized and clamped image tensor (B, C, H, W)
    """
    x = x.clone()

    if getattr(config.data, "normalize", None) == "resnet18_imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = x * std + mean

    return x.clamp(0.0, 1.0)

def _draw_cross_chw(image: torch.Tensor, y_center: int, x_center: int, color: torch.Tensor, cross_radius: int = 5) -> None:
    """
    Draws a cross at the specified center coordinates on a CHW image tensor.
    Inputs:
        - image: Image tensor in CHW format (3, H, W)
        - y_center: Y-coordinate of the center of the cross
        - x_center: X-coordinate of the center of the cross
        - color: Color tensor for the cross (3,)
        - cross_radius: Radius of the cross arms

    Outputs:
        - None (the image tensor is modified in place)
    """

    _, H, W = image.shape

    # Clamp center coordinates to image bounds and convert to int
    y_center = int(max(0, min(H - 1, y_center)))
    x_center = int(max(0, min(W - 1, x_center)))

    # Calculate the bounding coordinates for the cross arms
    y_0 = max(0, y_center - cross_radius)
    y_1 = min(H - 1, y_center + cross_radius)
    x_0 = max(0, x_center - cross_radius)
    x_1 = min(W - 1, x_center + cross_radius)

    # Draw vertical and horizontal lines to form a cross
    image[:, y_0:y_1 + 1, x_center] = color.view(3, 1)   # Vertical line
    image[:, y_center, x_0:x_1 + 1] = color.view(3, 1)   # Horizontal line

    return None

@torch.no_grad()
def visualize_pred_vs_gt(model: torch.nn.Module, loader: DataLoader, config: RunConfig, images2visualize: int = 4) -> torch.Tensor:
    
    # Use device from config
    device = torch.device(config.train.device)
    # Set model to evaluation mode
    model.eval()

    
    x, gridIndices_gt, bbox_gt_norm, class_gt = next(iter(loader))  # Get a single batch from the DataLoader
    x = x.to(device)                                                # Move data to the chosen device
    gridIndices_gt = gridIndices_gt.to(device)                      # Move ground truth grid indices to device

    center_pred, box_pred, cls_pred = model(x)                      # Forward pass through the model to get predictions

    B, _, H, W_image = x.shape                                      # Get batch size and image dimensions
    stride_S = int(getattr(config.grid, "stride_S", 1))             # Get stride from config, default to 1 if not specified
    stride_S = max(stride_S, 1)                                     # Ensure stride is at least 1

    # Predicted center cell via argmax
    center_flat = center_pred[:, 0].reshape(B, -1)                  # Flatten center predictions (B, H*W)
    center_flat_index = torch.argmax(center_flat, dim=1)            # Get index of max value (predicted center cell) for each batch element (B,)
    W_grid = int(config.grid.W)                                     # Get grid width from config
    hat_i = center_flat_index // W_grid                             # Calculate row index from flat index
    hat_j = center_flat_index % W_grid                              # Calculate column index from flat index

    # Denormalize images for visualization
    x_vis = _denormalize_for_tb(x, config).detach().cpu()

    # Colors (RGB) for markers
    color_gt = torch.tensor([0.0, 1.0, 0.0])  # Green
    color_predicted = torch.tensor([1.0, 0.0, 0.0])  # Red

    number_images = min(images2visualize, B)                        # Number of images to visualize
    imgs = []                                                       # Initialize list to hold images with drawn crosses

    # Draw crosses on images
    for b in range(number_images):
        image = x_vis[b].clone()  # (3,H,W)

        # GT center pixel from grid cell
        ig, jg = int(gridIndices_gt[b, 0].item()), int(gridIndices_gt[b, 1].item())
        y_center_gt = ig * stride_S + stride_S // 2
        x_center_gt = jg * stride_S + stride_S // 2

        # Pred center pixel from grid cell
        predictedLabel_i, predictedLabel_j = int(hat_i[b].item()), int(hat_j[b].item())
        y_center_pd = predictedLabel_i * stride_S + stride_S // 2
        x_center_pd = predictedLabel_j * stride_S + stride_S // 2

        # Draw crosses via helper function
        _draw_cross_chw(image, y_center_gt, x_center_gt, color_gt, cross_radius=6)
        _draw_cross_chw(image, y_center_pd, x_center_pd, color_predicted, cross_radius=6)

        imgs.append(image)

    grid = vutils.make_grid(imgs, nrow=2, padding=2)                  # Create a grid of images with crosses
    return grid

# Visualize inference result for a single image
def visualize_single_inference(
    img_path: str,
    pred: dict,
    stride_S: int,
    imgsz: int,
    class_names: list[str] | None = None,
    save_path: str | None = None,
):
    """
    Visualize inference result for a single image.

    Args:
        img_path: Path to input image
        pred: Output dict from decode_single
        stride_S: Grid stride in pixels
        imgsz: Input image size used during inference
        class_names: Optional list of class names
        save_path: If given, saves figure instead of showing it
    """

    # Load image (same resizing as inference)
    img = Image.open(img_path).convert("RGB")
    img = img.resize((imgsz, imgsz))

    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(img)

    # Grid indices
    i_hat, j_hat = pred["grid_Indices_hat"]
    i_hat = int(i_hat)
    j_hat = int(j_hat)

    # Grid cell center in pixel coordinates
    cx = j_hat * stride_S + stride_S / 2
    cy = i_hat * stride_S + stride_S / 2

    # Box decoding
    dx, dy, w, h = pred["box_hat_list"]

    # Adjust center coordinates with offsets
    cx = cx + dx * stride_S
    cy = cy + dy * stride_S

    # Box width and height in pixels scaled to image size
    w_px = w * imgsz
    h_px = h * imgsz

    x1 = cx - w_px / 2
    y1 = cy - h_px / 2

    # Draw bounding box
    boundingBox = patches.Rectangle(
        (x1, y1),
        w_px,
        h_px,
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    )
    ax.add_patch(boundingBox)

    # Draw center point
    ax.plot(cx, cy, "ro")   # Red dot at center

    # Label
    cls = pred["cls_hat"]
    score = pred["center_score"]

    if class_names:
        label = f"{class_names[cls]} | {score:.2f}"
    else:
        label = f"Class {cls} | {score:.2f}"

    # Draw label above bounding box 
    ax.text(
        x1,
        y1 - 5,
        label,
        color="red",
        fontsize=10,
        bbox=dict(facecolor="black", alpha=0.5, pad=2)
    )

    ax.set_axis_off()   # Hide axes

    # Save or show figure
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.show()
