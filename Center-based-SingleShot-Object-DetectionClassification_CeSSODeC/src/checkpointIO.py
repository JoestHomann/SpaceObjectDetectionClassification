# checkpointIO.py
# Checkpoint save/load utilities for training and inference for Center-based Single Shot Object Detection and Classification (CeSSODeC).
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
# Author:                   J. Homann, C. Kern
# Email:                    st171800@stud.uni-stuttgart.de
# Created:                  24-Jan-2026 14:45:00
# References:
#   None
#
# Revision history:
#   None
#
# Implemented in VSCode 1.108.1
# 2026 in the Applied Machine Learning Course Project


from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

def save_checkpoint(path: Union[str, Path], model: nn.Module, optimizer: torch.optim.Optimizer, meta: Dict[str, Any]) -> None:
    """
    Saves the model and optimizer state along with metadata to a checkpoint file.
    
    Inputs:
        path: Path to save the checkpoint (e.g. path to last.pt)
        model: The model to save
        optimizer: The optimizer to save
        meta: Metadata (epoch, metrics, configs) to save in the checkpoint
    
    Outputs:
        None

    Stored in the checkpoint:
        - model_state
        - optimizer_state
        - meta
    """
    # Ensure the directory exists otherwise create it
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Data to be saved in the checkpoint
    # As checkpointData is of multiple types, we use typing.Dict and typing.Any
    checkpointData: Dict[str, Any] = {
        "model_state" : model.state_dict(),
        "optimizer_state" : optimizer.state_dict(),
        "meta" : meta,
    }

    # Save the checkpoint
    torch.save(checkpointData, checkpoint_path)

def load_checkpoint(path: Union[str, Path], model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, map_location: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Loads the model and optimizer state from a checkpoint file. Optimizer and map_location are 
    optional as they are not needed for inference but for resuming training.

    Inputs:
        path: Path to the checkpoint file
        model: The model to load the state into
        optimizer: The optimizer to load the state into
        map_location: Device to map the checkpoint to (e.g. 'cpu' or 'cuda', see config.device)

    Outputs:
        meta: Metadata dictionary loaded from the checkpoint

    """
    # TODO: Outsource path validation to a separate function?

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load the checkpoint data
    checkpointData = torch.load(str(checkpoint_path), map_location=map_location)

    # Validate checkpoint data format
    if not isinstance(checkpointData, dict):
        raise ValueError(f"Invalid checkpoint format (expected dict): {checkpoint_path}")

    # Get model state from checkpointData
    model_state = checkpointData.get("model_state", None)
    if model_state is None:
        raise ValueError(f"Checkpoint missing key 'model_state': {checkpoint_path}")

    # Load model state into the model
    model.load_state_dict(model_state)

    # Get optimizer state from checkpointData
    optimizer_state = checkpointData.get("optimizer_state", None)
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    # Get metadata from checkpointData
    meta = checkpointData.get("meta", {})
    if meta is None:
        meta = {}   # Default to empty dict if no metadata found
    if not isinstance(meta, dict):  # Validate meta format/type
        raise ValueError(f"Invalid 'meta' type (expected dict): {checkpoint_path}")

    
    return meta