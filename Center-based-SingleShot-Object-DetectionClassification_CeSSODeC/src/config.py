# config.py
# Configuration file for Center-based Single Shot Object Detection and Classification (CeSSODeC)
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
# Created:                  23-Jan-2026 10:00:00
# References:
#   None
#
# Revision history:
#   - Added LossConfig class for loss-related hyperparameters (28-Jan-2026, J. Homann, C. Kern)
#
# Implemented in VSCode 1.108.1
# 2026 in the Applied Machine Learning Course Project

from dataclasses import dataclass

@dataclass(frozen=True)
class GridConfig:

    """
    Configuration class for grid settings in CeSSODeC.

    Attributes:
    imgsz: input image size in pixels e.g. imgsz = 640 means 640x640 pixels
    stride_S: feature stride in pixels (How big the object is compared to the image size) 
              e.g. stride_S = 8 means that the feature map is 1/8th the size of the input image
    H: grid height (imgsz / stride_S)
    W: grid width (imgsz / stride_S) (for now H = W)
    """
    imgsz: int = 320
    stride_S: int = 32 # 32 because of ResNet18   # TODO: Rename to featureStride_grid?
    H: int = imgsz // stride_S
    W: int = imgsz // stride_S
    
@dataclass(frozen=True)
class LossConfig:
    """
    Loss configuration for CeSSODeC.
    
    Attributes:
        gaussHm_sigma: float
            Standard deviation for Gaussian heatmap generation.
        BCE_scale: float
            Scaling factor for Binary Cross Entropy loss component.
    """
    gaussHm_sigma: float = 2.0
    BCE_scale: float = 10.0

@dataclass(frozen=True)
class DataConfig:
    """
    Dataset configuration for CeSSODeC.
    
    Attributes:

        root: str
            Dataset root directory path.
        normalize: str
            Normalzation mehtod for input images (Resnet, imagenet).
            First try: Imagenet
            Possible improvements: Dataset specific normalization
        num_classes: int
            Amount of classes in the dataset.  
    """
    datasetRoot: str = ""
    normalize: str = "resnet18_imagenet"
    num_classes: int = 11       # TODO: Do we need this?? Already defined in class_names.txt?

@dataclass(frozen=True)   
class ModelConfig:
    """
    Model configuration for CeSSODeC.

    Attributes:
    backbone: str
        Backbone architecture for feature extraction.
        First try: resnet18
        possible improvements: resnet34, resnet50 (hardware heavy)
    num_classes: int
        Amount of classes in the model. May differ from dataset num_classes if using different datasets for training and evaluation.
    feature_stride: int
        Feature stride for the model (should match DataConfig stride_S).
    """
    backbone: str = "resnet18_imagenet"
    num_classes: int = 11  
    feature_stride: int = 32    # TODO: Remove redundancy? # TODO: Rename to featureStride_model?


@dataclass(frozen=True)
class TrainConfig:
    """
    Training hyperparameters and configuration/settings for CeSSODeC.

    Attributes:
        device: "cuda" or "cpu"
        lr: learning rate of optimizer(e.g. Adam)
        weight_decay: weight decay for optimizer
        epochs: number of epochs that the model has to go through for training
        batch_size: Number of training samples (images) in one forward/backward pass
        num_workers: Number of DataLoaders, which load/prepare the samples to 
                     then feed these batches to the model during training
        activateAMP: Boolean to activate automatic mixed precision training to save memory 
                     and speed up training (use float16/bfloat32 instead of float32 everytime)
        ckpt_last_path: Path to save the last checkpoint of the model
        ckpt_best_path: Path to save the best checkpoint of the model (Usally the same as ckpt_last_path) # TODO: Remove?
        seed: (Random) seed for reproducibility
    """
    device: str = "cuda"
    lr: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 50
    batch_size: int = 32
    num_workers: int = 4
    activateAMP: bool = False
    ckpt_last_path: str = './runs/run1/checkpoints/last.pth'   # TODO: Update runX to dynamic run folder
    ckpt_best_path: str = './runs/run1/checkpoints/best.pth'
    seed: int = 69420

@dataclass(frozen=True)
class RunConfig:
    """
    Run configuration for CeSSODeC. This class puts all other configuration classes together. 
        
    SINGLE SOURCE OF TRUTH! (Donald J. Trump - The holy father of config classes)

    Inputs:
         data: DataConfig class
         grid: GridConfig class
         model: Modelconfig class
         train: TrainConfig class
         loss: LossConfig class
     """
    data: DataConfig
    grid: GridConfig
    model: ModelConfig
    train: TrainConfig
    loss: LossConfig
