# train_main.py
# Main training script to parse arguments for Center-based Single Shot Object Detection and Classification (CeSSODeC).
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
# Created:                  24-Jan-2026 16:10:00
# References:
#   None
#
# Revision history:
#   - Added loss parse arguments for hyperparameter optimization (28-Jan-2026, J. Homann, C. Kern)
#
# Implemented in VSCode 1.108.1
# 2026 in the Applied Machine Learning Course Project

import argparse
from config import DataConfig, GridConfig, ModelConfig, RunConfig, TrainConfig, LossConfig
from train import fit


def parse_args_tr() -> argparse.Namespace:
    """
    Parses command line arguments for training configuration. So that we can change config entries from command line.
    Inputs:
        None

    Outputs:
        args: argparse.Namespace object with training settings
    """

    parser = argparse.ArgumentParser(description="Train CeSSODeC model")

    # Dataset argument (required)
    parser.add_argument("--datasetRoot", type=str, required=True, help="Dataset root directory path containing images and labels folders")

    # Training hyperparameters (optional)
    # Defaults are "None" here as they will be set from config file (see below) in build_config()
    parser.add_argument("--epochs", type=int, required=False, default=None, help="Number of training epochs (overrides TrainConfig.epochs)")
    parser.add_argument("--batch_size", type=int, required=False, default=None, help="Batch size for training (overrides TrainConfig.batch_size)")
    parser.add_argument("--lr", type=float, required=False, default=None, help="Learning rate (overrides TrainConfig.lr)")
    parser.add_argument("--weight_decay", type=float, required=False, default=None, help="Weight decay (overrides TrainConfig.weight_decay)")
    parser.add_argument("--num_workers", type=int, required=False,default=None, help="DataLoader workers (overrides TrainConfig.num_workers)")
    parser.add_argument("--device", type=str, required=False,default=None, help='Device, e.g. "cuda" or "cpu" (overrides TrainConfig.device)')
    parser.add_argument("--amp", required=False, action="store_true", help="Enable/disable automatic mixed precision (overrides TrainConfig.activateAMP)")

    # Seed for Reproducibility (optional)
    parser.add_argument("--seed", type=int, required=False, default=None, help="Change random seed (overrides TrainConfig.seed)")

    # Checkpoint paths (optional)
    parser.add_argument("--checkpointPath_last", type=str, required=False, default=None, help="Change Directory/Filepath of the last.pt file (overrides TrainConfig.ckpt_last_path)")
    parser.add_argument("--checkpointPath_best", type=str, required=False, default=None, help="Change Directory/Filepath of the best.pt file (overrides TrainConfig.ckpt_best_path)")

    # Image and Object settings
    parser.add_argument("--imgsz", type=int, required=False, default=None, help="Change input images size (overrides GridConfig.imgsz)")
    parser.add_argument("--stride_S", type=int, required=False, default=None, help="Change feature stride (overrides GridConfig.stride_S)")
    parser.add_argument("--normalize", type=str, required=False, default=None, help="Normalization scheme (overrides DataConfig.normalize)")

    # HPO settings can be added here as needed
    parser.add_argument("--gaussHm_sigma", type=float, default=None, help="Change Gaussian heatmap sigma (overrides LossConfig.gaussHm_sigma)")
    parser.add_argument("--BCE_scale", type=float, default=None, help="Change BCE loss scale (overrides LossConfig.BCE_scale)")
    parser.add_argument("--run_name", type=str, default="default", help = "Name of the current run for logging purposes")

    return parser.parse_args()

def build_config_tr(ParserArguments: argparse.Namespace) -> RunConfig:
    """
    Builds the RunConfig instance/object by combining default configurations with command line arguments.

    Inputs:
        ParserArguments: argparse.Namespace object with parsed command line arguments
    
    Outputs:
        cfg: RunConfig instance/object with final training configuration

    """
    # Initialize instance/object of class GridConfig
    grid = GridConfig()

    # Overwrite the instances GridConfig values with the parsed arguments (if parsed) and store them in grid
    if ParserArguments.imgsz is not None or ParserArguments.stride_S is not None:
        imgsz = ParserArguments.imgsz if ParserArguments.imgsz is not None else grid.imgsz
        stride_S = ParserArguments.stride_S if ParserArguments.stride_S is not None else grid.stride_S
        H = imgsz // stride_S
        W = imgsz // stride_S
        grid = GridConfig(imgsz=imgsz, stride_S=stride_S, H=H, W=W)

    # Initialize instance/object of class DataConfig
    data = DataConfig(datasetRoot=ParserArguments.datasetRoot)
    
    # Overwrite the instances DataConfig values with the parsed arguments (if parsed) and store them in data
    if ParserArguments.normalize is not None:
        data = DataConfig(datasetRoot=ParserArguments.datasetRoot, normalize=ParserArguments.normalize, num_classes=data.num_classes)

    # Initialize instance/object of class ModelConfig
    model = ModelConfig()

    # Initialize instance/object of class TrainConfig
    train = TrainConfig(run_name=ParserArguments.run_name if ParserArguments.run_name is not None else "default")

    # Initialize instance/object of class LossConfig
    loss = LossConfig()
    
    # Overwrite the instances LossConfig values with the parsed arguments (if parsed) and store them in loss
    if ParserArguments.gaussHm_sigma is not None or ParserArguments.BCE_scale is not None:
        loss = LossConfig(
            gaussHm_sigma=ParserArguments.gaussHm_sigma 
                if ParserArguments.gaussHm_sigma is not None else loss.gaussHm_sigma,
            BCE_scale=ParserArguments.BCE_scale 
                if ParserArguments.BCE_scale is not None else loss.BCE_scale,
        )

    # Overwrite the instances TrainConfig values with the parsed arguments (if parsed) and store them in train
    if ParserArguments.epochs is not None:
        epochs = ParserArguments.epochs
    if ParserArguments.batch_size is not None:
        batch_size = ParserArguments.batch_size
    if ParserArguments.lr is not None:
        lr = ParserArguments.lr
    if ParserArguments.weight_decay is not None:
        weight_decay = ParserArguments.weight_decay
    if ParserArguments.num_workers is not None:
        num_workers = ParserArguments.num_workers
    if ParserArguments.device is not None:
        device = ParserArguments.device
    if ParserArguments.amp is not None:
        activateAMP = ParserArguments.amp
    if ParserArguments.seed is not None:
        seed = ParserArguments.seed
    if ParserArguments.checkpointPath_last is not None:
        ckpt_last_path = ParserArguments.checkpointPath_last
    if ParserArguments.checkpointPath_best is not None:
        ckpt_best_path = ParserArguments.checkpointPath_best
    train = TrainConfig(
        device=ParserArguments.device if ParserArguments.device is not None else train.device,
        lr=ParserArguments.lr if ParserArguments.lr is not None else train.lr,
        weight_decay=ParserArguments.weight_decay if ParserArguments.weight_decay is not None else train.weight_decay,
        epochs=ParserArguments.epochs if ParserArguments.epochs is not None else train.epochs,
        batch_size=ParserArguments.batch_size if ParserArguments.batch_size is not None else train.batch_size,
        num_workers=ParserArguments.num_workers if ParserArguments.num_workers is not None else train.num_workers,
        activateAMP=ParserArguments.amp if ParserArguments.amp is not None else train.activateAMP,
        seed=ParserArguments.seed if ParserArguments.seed is not None else train.seed,
        ckpt_last_path=ParserArguments.checkpointPath_last if ParserArguments.checkpointPath_last is not None else train.ckpt_last_path,
        ckpt_best_path=ParserArguments.checkpointPath_best if ParserArguments.checkpointPath_best is not None else train.ckpt_best_path,
    )

    return RunConfig(data=data, grid=grid, model=model, train=train, loss=loss)  # Combine all configurations into RunConfig instance/object and return it


# Main script execution point
if __name__ == "__main__":
    """
    Main execution point for training CeSSODeC model.

    Inputs:
        None   

    Outputs:
        None
    """
    arguments = parse_args_tr()                 # Reads the command-line inputs (e.g. --epochs) and stores them in an the argparse.Namespace

    config = build_config_tr(arguments)         # Writes those parsed command line values into your structured RunConfig (with DataConfig, 
                                                # GridConfig, ModelConfig, TrainConfig). If a command line option is None (so not provided),
                                                # it keeps the default from the config

    fit(config)                                 # Start training with the final configuration


