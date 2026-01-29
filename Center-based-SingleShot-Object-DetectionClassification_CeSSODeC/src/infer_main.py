# infer_main.py
# Main inference script for Center-based Single Shot Object Detection and Classification (CeSSODeC).
# Command line arguments are parsed here and passed to the inference function.
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
# Created:                  24-Jan-2026 18:30:00
# References:
#   None
#
# Revision history:
#   None
#
# Implemented in VSCode 1.108.1
# 2026 in the Applied Machine Learning Course Project


# infer_main.py
from __future__ import annotations

# CLI entrypoint for inference (single image).

import argparse
from pathlib import Path

import torch

from config import DataConfig, GridConfig, ModelConfig, RunConfig, TrainConfig, LossConfig
from infer import load_model_for_inference, preprocess_image, decode_single

from visualizationHelpers import visualize_single_inference
from pathlib import Path
from infer import run_inference, InferConfig

def parse_args_inf() -> argparse.Namespace:

    """
    Parses command line arguments for inference configuration.
    
    Inputs:
        None

    Outputs:
        args: argparse.Namespace object with inference settings
    """

    parser = argparse.ArgumentParser(description="Inference configuration for CeSSODeC")
    
    # Model checkpoint path (required)
    parser.add_argument("--checkpointPath", type=str, required=True, help="Path to checkpoint file (e.g. /your/folder/last.pt)")

    # Input image path (required)
    parser.add_argument("--imagePath", type=str, required=True, help="Path to input image file (e.g. /your/folder/image069420.jpg)")

    # Other overrides (optional)
    parser.add_argument("--device", type=str, required=False, default=None, help='Device for inference, e.g. "cuda" or "cpu" (overrides ModelConfig.device)')
    parser.add_argument("--normalize", type=str, required=False, default=None, help='Normalization method, e.g. "imagenet" or "none" (overrides ModelConfig.normalize)')
    parser.add_argument("--imgsz", type=int, required=False, default=None, help="Input image size in pixels (overrides GridConfig.imgsz)")
    parser.add_argument("--stride_S", type=int, required=False, default=None, help="Feature stride in pixels (overrides GridConfig.stride_S)")

    return parser.parse_args()

def build_config_inf(ParserArguments: argparse.Namespace) -> RunConfig:
    """
    Builds the RunConfig for inference based on command line arguments.
    
    Inputs:
        args: argparse.Namespace object with inference settings
    Outputs:
        cfg: RunConfig object with structured configuration
    """
    # Initialize instance/object of class GridConfig
    grid = GridConfig()

    # Overwrite the instances GridConfig values with the parsed arguments (if parsed) and store them in grid
    if ParserArguments.imgsz is not None or ParserArguments.stride_S is not None:
        imgsz = ParserArguments.imgsz if ParserArguments.imgsz is not None else grid.imgsz
        stride_S = ParserArguments.stride_S if ParserArguments.stride_S is not None else grid.stride_S

        # Validate divisibility
        if imgsz % stride_S != 0:
            raise ValueError(f"imgsz ({imgsz}) must be divisible by stride_S ({stride_S})")
        
        H = imgsz // stride_S       # Height in feature map
        W = imgsz // stride_S       # Width in feature map
        grid = GridConfig(imgsz=imgsz, stride_S=stride_S, H=H, W=W)     # Update grid with new values
    
    # Initialize instance/object of class DataConfig
    data = DataConfig(datasetRoot="")  # Dataset root not needed for inference

    # Overwrite the instances DataConfig values with the parsed arguments (if parsed) and store them in data
    if ParserArguments.normalize is not None:
        data = DataConfig(datasetRoot="", normalize=ParserArguments.normalize, num_classes=data.num_classes)
    
    # Initialize instance/object of class ModelConfig
    model = ModelConfig()

    # Initialize instance/object of class TrainConfig
    train = TrainConfig()
    # Overwrite the instances TrainConfig values with the parsed arguments (if parsed) and store them in train
    if ParserArguments.device is not None:
        device = ParserArguments.device
        train = TrainConfig(device=device) 

    # Initialize instance/object of class LossConfig
    loss = LossConfig() 

    # Return the combined RunConfig
    return RunConfig(data=data, grid=grid, model=model, train=train, loss=None)  # LossConfig not needed for inference



def main() -> None:
    """
    Main function for inference script. Builds configuration with parsed command line arguments,
    loads the model, and runs inference on the specified image(s).
    
    Inputs:
        None

    Outputs:
        None
    """
    parsedArguments = parse_args_inf()                  # Reads the command-line inputs (e.g. --checkpointPath) and stores them in an the argparse.Namespace
    config_inf = build_config_inf(parsedArguments)      # Writes those parsed command line values into the RunConfig (with DataConfig, GridConfig, ModelConfig, TrainConfig).
                                                        # If a command line option is None (so not provided), it keeps the default from the config

    img_path = Path(parsedArguments.imagePath)          # Path to input image file, parsed via --imagePath command line argument

    # Determine if input is a file or directory and gather image paths
    if img_path.is_file():                              # Single image file -> only one image    
        image_paths = [str(img_path)]                   # Single image path
    elif img_path.is_dir():                             # Directory of images -> gather all image files in directory
        image_paths = sorted([
            str(p) for p in img_path.iterdir()          # Iterate over files in directory
            if p.suffix.lower() in [".jpg"]             # Filter for .jpg files only
        ])
        if len(image_paths) == 0:                       # If no images found in directory, raise error
            raise RuntimeError(f"No images found in directory: {img_path}")
    else:
        raise FileNotFoundError(f"Invalid imagePath: {img_path}")   # If invalid path provided, raise error   

    device = torch.device(config_inf.train.device) # Initialize device

    # Load the trained model for inference
    model = load_model_for_inference(
    ckpt_path=parsedArguments.checkpointPath,
    model_cfg=config_inf.model,
    grid_cfg=config_inf.grid,
    device=config_inf.train.device
)
  
    
    # Load inference configuration
    infer_cfg = InferConfig(
        device=config_inf.train.device,
        normalize=config_inf.data.normalize
    )

    # Run inference on the input images
    results = run_inference(
        ckpt_path=parsedArguments.checkpointPath,
        inputs=image_paths,
        model_cfg=config_inf.model,
        grid_cfg=config_inf.grid,
        infer_cfg=infer_cfg,
    )

    # Create output directory for inference results if it doesn't exist yet
    output_dir = Path("infer_runs") / "Images_predict"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save inference results for each image and visualize them
    for res in results:
        img_name = Path(res["input_path"]).name
        save_path = output_dir / f"inference_result_{img_name}"

        # Visualize via helper function
        visualize_single_inference(
            img_path=res["input_path"],
            pred=res,
            stride_S=config_inf.grid.stride_S,
            imgsz=config_inf.grid.imgsz,
            save_path=str(save_path)
        )   




if __name__ == "__main__":
    main()
