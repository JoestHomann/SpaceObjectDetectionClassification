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

#from cmath import rect
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math


from config import RunConfig

from PIL import Image
import matplotlib.patches as patches



# def plotHeatmap # TODO: Implement heatmap plotting function
# def plotBoundingBoxes # TODO: Implement bounding box plotting function


# Plotting function for confusion matrix
def plotConfMatrix(confusion_matrix: np.ndarray, class_names: list[str]):
    """
    Plots a confusion matrix using Matplotlib.

    Args:
        confusion_matrix (np.ndarray): The confusion matrix to plot.
        class_names (list[str]): List of class names corresponding to the matrix indices.

    Returns:
        plt.Figure: The Matplotlib figure containing the confusion matrix plot.
    """
    confusionMatrix = confusion_matrix.astype(np.float64)

    # Always normalize the confusion matrix
    row_sums = confusionMatrix.sum(axis=1, keepdims=True)               # Sum of each row (true labels)
    confusionMatrix = confusionMatrix / np.clip(row_sums, 1.0, None)    # Normalize each row to sum to 1, avoid division by zero by clipping
    confusionMatrix = confusionMatrix.T                                 # Transpose for correct orientation (so its like the YOLO conf matrix)

    figure, axes = plt.subplots(figsize=(8, 8))                 # Set figure size and axes
    imageObject = axes.imshow(confusionMatrix,                  # Display the confusion matrix as an image 
                            interpolation='nearest',            # Use nearest neighbor interpolation
                            cmap = plt.colormaps["Blues"],      # Use blue colormap
                            vmin=0.0, vmax=1.0                  # Set color scale from 0 to 1
                               )         

    axes.set_title("Confusion Matrix (normalized)")             # Set title of the plot
    figure.colorbar(imageObject, ax=axes, fraction=0.046, pad=0.04)  # Add colorbar to the plot

    tick_marks = np.arange(len(class_names))                     # Create tick marks for each class
    axes.set_xticks(tick_marks)                                  # Set x-ticks
    axes.set_yticks(tick_marks)                                  # Set y-ticks
    axes.set_xticklabels(class_names, rotation=45, ha="right")   # Set x-tick labels with rotation
    axes.set_yticklabels(class_names)                            # Set y-tick labels

    axes.set_xlabel("True label")                                # Set x-axis label
    axes.set_ylabel("Predicted label")                           # Set y-axis label

    # Write values/probabilities to cells
    for i in range(confusionMatrix.shape[0]):
        for j in range(confusionMatrix.shape[1]):
            value = confusionMatrix[i, j]
            if value > 0.0:
                axes.text(j, i, f"{value:.2f}", ha="center", va="center", color="black")
    

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

# Internal helper function to convert (cx, cy, w, h) to (x1, y1, x2, y2)
def _xywh_to_xyxy(x_center_pred: float, y_center_pred: float, w_pred: float, h_pred: float) -> tuple[float, float, float, float]:
    """
    Converts bounding box from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2).

    Inputs:
        - cx: Center x-coordinate
        - cy: Center y-coordinate
        - w: Width of the bounding box
        - h: Height of the bounding box

    Outputs:
        - x1, y1, x2, y2: Corner coordinates of the bounding box
    """
    x1_pred = x_center_pred - 0.5 * w_pred
    y1_pred = y_center_pred - 0.5 * h_pred
    x2_pred = x_center_pred + 0.5 * w_pred
    y2_pred = y_center_pred + 0.5 * h_pred
    return x1_pred, y1_pred, x2_pred, y2_pred

def _to_uint8_chw(image2drawOn_chw: torch.Tensor) -> torch.Tensor:
    return (image2drawOn_chw * 255.0).clamp(0.0, 255.0).to(torch.uint8)

# Visualize predictions vs. ground truth for a batch of images
@torch.no_grad()
def visualize_pred_vs_gt(
    model: torch.nn.Module,
    loader: DataLoader,
    config: RunConfig,
    images2visualize: int = 64,
    class_names: list[str] | None = None
) -> torch.Tensor:

    """
    Visualizes predictions vs. ground truth for a batch of images.

    Inputs:
        model: The trained model for inference
        loader: DataLoader providing batches of images and ground truth
        config: RunConfig object with configuration settings
        images2visualize: Number of images to visualize from the batch
        class_names: Optional list of class names for labeling
    Outputs:
        grid: A grid tensor containing the visualized images with predictions and ground truth

    """
    # Get device from config
    device = torch.device(config.train.device)
    # Set model to evaluation mode
    model.eval()

    # Get a single batch from the DataLoader
    inputBatchImages, gridIndices_gt, bbox_gt_norm, class_gt = next(iter(loader))   # Get one batch of data
    inputBatchImages = inputBatchImages.to(device)                                  # Move input images to device
    gridIndices_gt = gridIndices_gt.to(device)                                      # Move ground truth grid indices to device
    bbox_gt_norm = bbox_gt_norm.to(device)                                          # Move ground truth bounding boxes to device
    class_gt = class_gt.to(device)                                                  # Move ground truth classes to device

    # Forward pass through the model to get predictions
    center_pred, box_pred, cls_pred = model(inputBatchImages)

    # Get batch size and image dimensions
    B, _, H_images, W_images = inputBatchImages.shape
    stride_S = int(getattr(config.grid, "stride_S", 1))
    stride_S = max(stride_S, 1)

    # Predicted center cell via argmax
    center_flat = center_pred[:, 0].reshape(B, -1)
    center_flat_index = torch.argmax(center_flat, dim=1)

    # Convert flat index to 2D grid indices
    W_grid = int(config.grid.W)
    i_pred = center_flat_index // W_grid
    j_pred = center_flat_index % W_grid

    # Denormalize images for visualization (CPU, 0..1)
    inputBatchImages_denormalized = _denormalize_for_tb(inputBatchImages, config).detach().cpu()

    # Set olors for drawing
    color_gt = torch.tensor([0.0, 1.0, 0.0])    # Green for ground truth
    color_pred = torch.tensor([1.0, 0.0, 0.0])  # Red for prediction

    # Get number of images to visualize from input argument or take full batch when batch is smaller than images2visualize
    number_images = min(images2visualize, B)
    imagesList = [] # List to hold images with drawn boxes

    for image_i in range(number_images):
        image2drawOn = inputBatchImages_denormalized[image_i].clone()  # (3,H,W) float in [0,1]

        # Get ground truth box in pixel coordinates
        xc_gt_norm, yc_gt_norm, w_gt_norm, h_gt_norm = bbox_gt_norm[image_i].detach().cpu().tolist()

        x_center_gt = xc_gt_norm * W_images
        y_center_gt = yc_gt_norm * H_images
        w_gt = w_gt_norm * W_images
        h_gt = h_gt_norm * H_images
        x1_gt, y1_gt, x2_gt, y2_gt = _xywh_to_xyxy(x_center_gt, y_center_gt, w_gt, h_gt)

        # Get predicted box in pixel coordinates
        i_hat = int(i_pred[image_i].item())     # Row index of predicted cell for image i
        j_hat = int(j_pred[image_i].item())     # Column index of predicted cell for image i

        # Grid cell center in pixel coordinates
        x_center_cell = j_hat * stride_S + stride_S / 2.0   # Column index to x-coordinate
        y_center_cell = i_hat * stride_S + stride_S / 2.0   # Row index to y-coordinate

        # Pred box at predicted cell
        # dx = offset in x direction (normalized to stride_S)
        # dy = offset in y direction (normalized to stride_S)
        # w_pred_norm = width prediction (normalized to image width)
        # h_pred_norm = height prediction (normalized to image height)
        dx, dy, w_pred_norm, h_pred_norm = box_pred[image_i, :, i_hat, j_hat].detach().cpu().tolist()

        # Adjust center coordinates with offsets
        x_center_pred = x_center_cell + dx * stride_S
        y_center_pred = y_center_cell + dy * stride_S
        w_pred = w_pred_norm * W_images
        h_pred = h_pred_norm * H_images

        # Convert from (x_center, y_center, w, h) to (x1, y1, x2, y2)
        x1_pred, y1_pred, x2_pred, y2_pred = _xywh_to_xyxy(x_center_pred, y_center_pred, w_pred, h_pred)

        # Clip to image bounds
        x1_gt = max(0.0, min(x1_gt, W_images - 1.0)); x2_gt = max(0.0, min(x2_gt, W_images - 1.0))
        y1_gt = max(0.0, min(y1_gt, H_images - 1.0)); y2_gt = max(0.0, min(y2_gt, H_images - 1.0))
        x1_pred = max(0.0, min(x1_pred, W_images - 1.0)); x2_pred = max(0.0, min(x2_pred, W_images - 1.0))
        y1_pred = max(0.0, min(y1_pred, H_images - 1.0)); y2_pred = max(0.0, min(y2_pred, H_images - 1.0))

        # Pred class and probabilities at predicted cell
        cls_logits_cell = cls_pred[image_i, :, i_hat, j_hat].detach().cpu()     # Get class logits at predicted cell
        class_hat = int(torch.argmax(cls_logits_cell).item())                   # Predicted class index at predicted cell
        class_prob = float(F.softmax(cls_logits_cell, dim=0)[class_hat].item()) # Class probability at predicted cell

        # Center probability at predicted cell
        center_val = float(center_pred[image_i, 0, i_hat, j_hat].detach().cpu().item()) # Center logit at predicted cell
        center_prob = center_val if (0.0 <= center_val <= 1.0) else float(torch.sigmoid(torch.tensor(center_val)).item()) # Convert logit to probability

        # Calculate combined probability
        combined_prob = center_prob * class_prob

        # Get class name
        if class_names is not None and 0 <= class_hat < len(class_names):
            cls_name = class_names[class_hat]
        else:
            cls_name = f"class_{class_hat}"

        # Draw boxes with labels on image
        img_u8 = _to_uint8_chw(image2drawOn)

        # Prepare boxes and labels for drawing
        boxes = torch.tensor(
            [[x1_gt, y1_gt, x2_gt, y2_gt],
             [x1_pred, y1_pred, x2_pred, y2_pred]],
            dtype=torch.float32
        )

        # Get ground truth class name
        gt_cls = int(class_gt[image_i].detach().cpu().item())
        if class_names is not None and 0 <= gt_cls < len(class_names):  # Valid class index 
            gt_name = class_names[gt_cls]                               # Get class name
        else:
            gt_name = f"class_{gt_cls}"                                 # Fallback to class index string

        labels = [
            f"True: {gt_name}",
            f"Pred: {cls_name} with p={combined_prob:.3f}",
        ]

        # Draw bounding boxes using torchvision utility
        img_u8 = vutils.draw_bounding_boxes(
            img_u8,
            boxes=boxes,
            labels=labels,
            colors=["green", "red"],
            width=2,
            font_size=10,
        )

        # Draw center points
        image2drawOn = (img_u8.float() / 255.0).clamp(0.0, 1.0)

        # Draw center crosses
        _draw_cross_chw(image2drawOn, int(y_center_gt), int(x_center_gt), color_gt, cross_radius=6)  # Draw ground truth center cross
        _draw_cross_chw(image2drawOn, int(y_center_pred), int(x_center_pred), color_pred, cross_radius=6)  # Draw predicted center cross
        imagesList.append(image2drawOn)

    nrow = int(np.ceil(math.sqrt(number_images)))   # Number of images per row in grid
    grid = vutils.make_grid(imagesList, nrow=nrow, padding=2)
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
    i_pred, j_pred = pred["grid_Indices_hat"]
    i_pred = int(i_pred)
    j_pred = int(j_pred)

    # Grid cell center in pixel coordinates
    x_center_cell = j_pred * stride_S + stride_S / 2
    y_center_cell = i_pred * stride_S + stride_S / 2

    # Box decoding
    dx, dy, w, h = pred["box_hat_list"]

    # Adjust center coordinates with offsets
    x_center_pred = x_center_cell + dx * stride_S
    y_center_pred = y_center_cell + dy * stride_S

    # Box width and height in pixels scaled to image size
    w_px = w * imgsz
    h_px = h * imgsz

    x1 = x_center_pred - w_px / 2
    y1 = y_center_pred - h_px / 2

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
    ax.plot(x_center_pred, y_center_pred, "ro")   # Red dot at center

    # Class label prediction
    cls = int(pred["cls_hat"])           # Predicted class index as integer

    # Center logit score
    score_logit = pred["center_score"]
    
    center_probability = torch.sigmoid(torch.tensor(score_logit)).item()  # Convert logit to probability

    class_logits = np.array(pred["cls_logits_list"], dtype=np.float64)  # Class logits as numpy array
    class_logits = class_logits - np.max(class_logits)                     # For numerical stability
    class_probabilities = np.exp(class_logits) / np.sum(np.exp(class_logits))      # Softmax to get class probabilities
    class_probability = float(class_probabilities[cls])                                          # Probability of predicted class

    combined_probability = center_probability * class_probability   # Combined score


    if class_names:
        label = f"{class_names[cls]} | {combined_probability:.3f}"
    else:
        label = f"Class {cls} | {combined_probability:.3f}"

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
