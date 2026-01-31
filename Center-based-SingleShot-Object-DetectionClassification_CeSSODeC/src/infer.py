# infer.py
# Loss computation for Center-based Single Shot Object Detection and Classification (CeSSODeC).
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
# Created:                  24-Jan-2026 16:55:00
# References:
#   None
#
# Revision history:
#   None
#
# Implemented in VSCode 1.108.1
# 2026 in the Applied Machine Learning Course Project

""" 
infer.py is the inference module wich translates model outputs into final object detections and classifications.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torchvision.transforms.functional as TF
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from config import GridConfig, ModelConfig
from visualizationHelpers import visualize_inference_with_heatmap

@dataclass(frozen=True)
class InferConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    normalize: str = "resnet18_imagenet"  # Options: "resnet18_imagenet", "none"
    
    # Number of top detections to consider (usually 1)
    topk: int = 1


def _imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    """ 
    Normalize image using ImageNet statistics. 
    ResNet expects inputs normalized this way.
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)
    return (x - mean) / std


def preprocess_image(img_path: str, imgsz: int, normalize: str) -> torch.Tensor:
    """ 
    Preprocess input image for model inference.
    Args:
        img_path (str): Path to the input image.
        imgsz (int): Target image size for the model.
        normalize (str): Normalization method ("imagenet" or "none").
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    p = Path(img_path)
    if not p.is_file():
        raise FileNotFoundError(f"Image file not found: {img_path}")

    with Image.open(p) as img:
        img = img.convert("RGB")
        img = img.resize((imgsz, imgsz))
        img_tensor = TF.to_tensor(img)  # Convert to tensor [0, 1]

        if normalize.lower() == "resnet18_imagenet":
            img_tensor = _imagenet_normalize(img_tensor)

        return img_tensor.unsqueeze(0)  # Add batch dimension


def load_model_for_inference(
        ckpt_path: str,
        model_cfg: ModelConfig,
        grid_cfg: GridConfig,
        device: str
) -> torch.nn.Module:
    """ 
    Load the trained model for inference.
    Args:
        ckpt_path (str): Path to the model checkpoint.
        model_cfg (ModelConfig): Model configuration.
        grid_cfg (GridConfig): Grid configuration.
        device (str): Device to load the model on ("cuda" or "cpu").
    Returns:
        torch.nn.Module: Loaded model ready for inference.
    """
    from model import CeSSODeCModel  # Local import to avoid circular dependencies

    # Initialize model architecture
    model = CeSSODeCModel(model_cfg=model_cfg, grid_cfg=grid_cfg)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)   # ckpt = checkpoint, weights_only=False to avoid torch.load error with newer PyTorch versions

    if isinstance(ckpt, dict):  # check for different checkpoint formats
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        else:
            state_dict = ckpt
    else:
        raise ValueError("Checkpoint format not recognized.")

    model.load_state_dict(state_dict, strict=True)  # Load model weights
    model.to(device)  # Move model to device
    model.eval()  # Set model to evaluation mode
    return model


def decode_single(
        center_preds: torch.Tensor,
        box_preds: torch.Tensor,
        class_preds: torch.Tensor,
) -> dict[str, Any]:
    """ 
    Decode model outputs for a single image.
    Inputs:
        center_preds (torch.Tensor): Center predictions from the model.
        box_preds (torch.Tensor): Box predictions from the model.
        class_preds (torch.Tensor): Class predictions
    """
    if center_preds.dim() != 4 or box_preds.dim() != 4 or class_preds.dim() != 4:   # check input dimensions
        raise ValueError("Input tensors must have 4 dimensions (B, C, H, W).")
    B, _, H, W = center_preds.shape  # batch size, height, width, _ gets ignored
    C = class_preds.shape[1]  # number of classes

    b = 0  # batch index
    center_map = center_preds[b, 0]  # (H, W)

    # Index of the highest center prediction
    flat_idx = torch.argmax(center_map).item()
    # Row index         / Needed to extract index from flattened tensor (essentially mapping 1D index back to 2D)
    i_hat = int(flat_idx // W)
    j_hat = int(flat_idx % W)   # Column index

    # Center score at the detected location
    center_score = float(center_preds[b, 0, i_hat, j_hat].item())
    # Class logits at the detected location
    cls_logits = class_preds[b, :, i_hat, j_hat]
    cls_hat = int(torch.argmax(cls_logits).item())  # Predicted class index
    # Box predictions at the detected location
    box_hat = box_preds[b, :, i_hat, j_hat]

    # Convert box predictions to list (easier to handle)
    box_hat_list = [float(v.item())for v in box_hat]
    cls_logits_list = [float(v.item())
                       for v in cls_logits]  # Convert class logits to list

    return {
        "grid_Indices_hat": torch.tensor([i_hat, j_hat], dtype=torch.int64),
        "cls_hat": cls_hat,
        "box_hat": box_hat,
        "box_hat_list": box_hat_list,
        "center_score": center_score,
        "cls_logits_list": cls_logits_list,
        "H": H,
        "W": W,
        "C": C
    }


def run_inference(
        ckpt_path: str,
        inputs: list[str],
        model_cfg: ModelConfig,
        grid_cfg: GridConfig,
        infer_cfg: InferConfig,
        output_dir: Optional[str] = None,

) -> list[dict[str, Any]]:
    """ 
    Run inference on a list of input images.
    Inputs:
        ckpt_path (str): Path to the model checkpoint.
        inputs (list[str]): List of input image paths.
        model_cfg (ModelConfig): Model configuration.
        grid_cfg (GridConfig): Grid configuration.
        infer_cfg (InferConfig): Inference configuration.
        output_dir (Optional[str]): Directory to save output results. If None, results are not saved.
    Returns:
        list[dict[str, Any]]: List of detection results for each input image.
    """
    device = InferConfig.device  # Determine device for inference
    model = load_model_for_inference(
        ckpt_path=ckpt_path,
        model_cfg=model_cfg,
        grid_cfg=grid_cfg,
        device=device
    )   # Load the trained model

    results: list[dict[str, Any]] = []  # List to store results for each image

    for img_path in inputs:
        img_tensor = preprocess_image(
            img_path=img_path,
            imgsz=grid_cfg.imgsz,
            normalize=infer_cfg.normalize
        )   # Preprocess the input image
        img_tensor = img_tensor.to(device)  # Move image tensor to device

        with torch.no_grad():
            center_preds, box_preds, class_preds = model(
                img_tensor)  # Forward pass through the model
        
        decoded = decode_single(
            center_preds=center_preds,
            box_preds=box_preds,
            class_preds=class_preds
        )   # Decode model outputs
        decoded["input_path"] = img_path
        results.append(decoded)

        if output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            save_path = str(out_dir / f"{Path(img_path).stem}_box_class_Hmap.png")
            visualize_inference_with_heatmap(
                img_path=img_path,
                pred=decoded,
                center_preds=center_preds,
                stride_S=grid_cfg.stride_S,
                imgsz=grid_cfg.imgsz,
                class_names=None,  # Optionally, provide class names here
                save_path=save_path,
                overlay_alpha=0.4,
            )

    return results

