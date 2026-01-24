# dataset.py
# Defines the dataset tensors/shapes that will be used in CeSSODeC.
# 
# Details:
#   None
# 
# Syntax:  
#   Attention here! 
#   H, W = image/grid height/width 
#   w, h = bounding box width/height
#   TODO: Update syntax???
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
# Created:                  23-Jan-2026 12:00:00
# References:
#   None
#
# Revision history:
#   None
#
# Implemented in VSCode 1.108.1
# 2026 in the Applied Machine Learning Course Project

# from dataclasses import dataclass  "may be needed in the future"

from config import DataConfig, GridConfig
from grid_mapping import yolo_norm_to_grid

import torch 
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from PIL import Image

from pathlib import Path

from typing import Tuple

class Sample:
    """ 
    One sample points to an image and its corresponding label file,
    e.g. imageName = image09871.jpg points to labelName = image09871.txt
    
    with: 
        img_path = data/spark-2022-stream-1/images/train/image09871.jpg
        label_path = data/spark-2022-stream-1/labels/image09871.txt
    """
    img_path: Path          # Initializes image_path as an object of type/class Path
    label_path: Path


class SingleObjectYoloDataset(Dataset):
    """
    Reading the dataset in Yolo format for single object detection.
    Converts, normalizes and scales the image data and labels to the required format.

    Attributes:
        data_cfg: DataConfig object with dataset configuration
        grid_cfg: GridConfig object with grid configuration
        data: 'train' or 'val' to indicate dataset split
        
    Methods:
        __len__: Returns the number of samples in the dataset
        __getitem__: Retrieves a sample from the dataset at the given index
        _build_index: Builds an internal index of all samples in the dataset
        _load_image: Loads and preprocesses the image from the given path
        _pil_to_chw_float: Converts a PIL image to a float32 tensor in CHW format
        _imagenet_normalize: Normalizes the input image tensor using ImageNet mean and std
        _load_label: Loads single-line YOLO labels from the given label file
        _make_targets: Converts label information to target tensors for training
    """
    def __init__(self, data_cfg: DataConfig, grid_cfg: GridConfig, data: str) -> None:
        """
        Inputs:
            data_cfg: DataConfig object with dataset configuration
            grid_cfg: GridConfig object with grid configuration
            data: 'train' or 'val' to indicate dataset split           # TODO: Split Training and Validation Data
        """
        super().__init__()          #Initialize the parent class (Dataset) / Uses the structure of the parent class

        self.data_cfg = data_cfg    #saves the data configuration to the class instance
        self.grid_cfg = grid_cfg
        self.data = data          #useless until we implement a data split

        self._index = []            # Initialize an empty list to store dataset samples
        self._build_index()         # Creates an internal index of all samples in the dataset

        if len(self._index) == 0:
            raise RuntimeError(f"No samples found for data ={data} in {data_cfg.datasetRoot}")  # Raise an error if no samples are found for the specified split

    def __len__(self) -> int:
        """ Returns the number of samples in the dataset """
        return len(self._index)


    def __getitem__(self, idx:int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        "gets" one sample from the dataset at the given index "idx".
        Returns the processed image tensor and the 3 target tensors for training.

        Inputs:
            idx: index of the sample to retrieve

        Outputs: 
            torch tensors representing the ground truths from dataset
        """
        img_path, label_path = self._index[idx]  # Get the image and label paths for the sample at index idx
        
        x = self._load_image(img_path)            # Load and preprocess the image
        cls, xc, yc, w, h = self._load_label(label_path) # Load the label data
        
        ij_gt, bbox_gt_norm, cls_gt = self._make_targets(cls, xc, yc, w, h) # create ground truth tensors from label data
        
        return x, ij_gt, bbox_gt_norm, cls_gt 

        
    def _build_index(self) -> None:
        """
        Builds an internal deterministic index of all samples (image_path, label_path) in the selected dataset split.
        """ 
        root = Path(self.data_cfg.datasetRoot)                    # Convert the root path from string to Path object

        img_dir = root / "images" / self.data             # Construct the image directory path
        label_dir = root / "labels" / self.data           # Construct the label directory path

        if not img_dir.is_dir() or not label_dir.is_dir():
            raise RuntimeError(f"Image or label directory does not exist: {img_dir}, {label_dir}")  # Raise an error if directories do not exist
        
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue 
            label_path = label_dir / (img_path.stem + ".txt") # Construct the corresponding label file path
            if not label_path.is_file():
                raise RuntimeError(f"Missing label file for image: {img_path}")  # Raise an error if the label file does not exist
        
        self._index.append((img_path, label_path))





    def _load_image(self, img_path: Path) -> torch.Tensor:
        """
        Loads and preprocesses the image from the given path.
        Converts the image to a float32 tensor in CHW format and normalizes it.
        
        Inputs:
            img_path: Path object pointing to the image file
        Outputs:
            x: float32 tensor of shape (3, H, W) with normalized pixel values
        """
        img = Image.open(img_path)  # Open the image using PIL Image

        x = self._pil_to_chw_float(img)  # Convert the PIL image to a float32 tensor in CHW format

        if self.data_cfg.normalize == 'imagenet':
            x = self._imagenet_normalize(x)  # Normalize the image tensor using ImageNet mean and std

        return x


    @staticmethod   # Used here as function below does not need access to class instance (self)
    def _pil_to_chw_float(img: Image.Image) -> "torch.Tensor":
        """
        Converts a PIL image to a float32 tensor in CHW format with pixel values in [0, 1].
        
        Inputs:
            img: PIL Image object

        Outputs:
            x: float32 tensor of shape (3, H, W) with pixel values in [0, 1]
        """
        img = img.convert("RGB")  # Ensure image is in RGB format
        
        x = TF.to_tensor(img)  # Convert PIL image to tensor

        return x


    @staticmethod   # Used here as function below does not need access to class instance (self)
    def _imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the input image tensor x using ImageNet mean and std (pre-defined).
        See documentation for details (https://docs.pytorch.org/vision/0.8/models.html).
        TODO: Implement dataset specific mean and std normalization for possible improvements?
        
        Inputs:
            x: float32 tensor of shape (3, H, W) with pixel values in [0, 1]

        Outputs:
            x_norm: float32 tensor of shape (3, H, W) with normalized pixel values
        """
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3,1,1)  # ImageNet mean for each channel
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3,1,1)   # ImageNet std for each channel

        x_norm = (x - imagenet_mean) / imagenet_std  # Normalize the image tensor

        return x_norm



    def _load_label(self, label_path: Path) -> tuple[int, float, float, float, float]:
        """
        Loads single-line YOLO labels from the given label file (image[imageNumber].txt)
        
        Inputs:
            label_path: Path object pointing to the label file

        Outputs:
            cls: class index of the object
            xc: normalized x center coordinate [0, 1]
            yc: normalized y center coordinate [0, 1]
            w: normalized width [0, 1]
            h: normalized height [0, 1]
        
        """

        labelFileContent = label_path.read_text(encoding="utf-8").strip()  # Read the content of the label file as a string and remove leading/trailing whitespace
        
        #TODO: Add error handling for empty or malformed label files

        cls_str, xc_str, yc_str, w_str, h_str = labelFileContent.split()  # Split the content into individual components

        return int(cls_str), float(xc_str), float(yc_str), float(w_str), float(h_str)  # Convert the string components to appropriate types and return them

    def _make_targets(self, cls: int, xc_norm: float, yc_norm:float, w: float, h: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Converts one objects label information from YOLO format to 3 target tensors for training. Each tensor is used to train one part of the model:
        1. Grid indices (i,j) for centerpoint prediction (Where is the object? So which cell to supervise in the training)
        2. Normalized bounding box (xc, yc, w, h) for bounding box regression (How big is the bounding box of the object? 
        And where is it precisely located? Precise location is needed for bounding box training -> "sub-cell precision")
        3. Class index used for classification at the supervised cell (What type of object is it?)
        
        Inputs:
            cls: class index of the object
            xc: normalized x center coordinate [0, 1]
            yc: normalized y center coordinate [0, 1]
            w: normalized width [0, 1]
            h: normalized height [0, 1]
        
        Outputs:
            ij_gt: int64 tensor of shape (2,) with grid indices (i,j)
            bbox_gt_norm: float32 tensor, shape (4,) with normalized bounding box (xc, yc, w, h)
            cls_gt: int64 tensor, scalar shape (), class index

        Annotation: gt = ground truth (true label)
        """

        # Map center to grid index
        i, j = yolo_norm_to_grid(xc_norm=xc_norm, yc_norm=yc_norm, grid=self.grid_cfg)

        # Create target tensors by combining each type of data (grid_indices, normalized center coordinates, class) to one tensor
        gridIndices_gt = torch.tensor([i, j], dtype=torch.int64)                    # Grid indices (i,j)
        bbox_gt_norm = torch.tensor([xc_norm, yc_norm, w, h], dtype=torch.float32)  # Normalized bounding box
        cls_gt = torch.tensor(cls, dtype=torch.int64)                               # Class index

        return gridIndices_gt, bbox_gt_norm, cls_gt