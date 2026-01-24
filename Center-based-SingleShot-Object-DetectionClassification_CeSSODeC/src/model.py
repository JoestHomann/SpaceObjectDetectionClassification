# model.py
#  Defines the neural network model architecture for CeSSODeC. So far it implements
#  a ResNet-based backbone with 3 convolutional 1x1 heads for center,
#  box and class predictions.
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
# Created:                  23-Jan-2026 13:00:00
# References:
#   None
#
# Revision history:
#   None
#
# Implemented in VSCode 1.108.1
# 2026 in the Applied Machine Learning Course Project


import torch
import torch.nn as nn
import torchvision.models as models

from config import GridConfig, ModelConfig


class CeSSODeCModel(nn.Module):
    """
    Center-based Single Shot Object Detection and Classification (CeSSODeC) Model.
    
    This model uses a ResNet backbone and three separate 1x1 convolutional heads for
    predicting object centers, bounding box dimensions, and class probabilities.
    

    """

    def __init__(self, model_cfg: ModelConfig, grid_cfg: GridConfig) -> None:
        """
        Initializes the CeSSODeC model. Sets up the backbone and prediction heads.
        
        Inputs:
            model_cfg: ModelConfig object with model configuration
            grid_cfg: GridConfig object with grid configuration
        
        Outputs:
            None

        """
        super().__init__()

        # Store configurations in object/instance
        self.model_cfg = model_cfg
        self.grid_cfg = grid_cfg

        # Build the ResNet18-based backbone with pre-trained weights from ImageNet
        # TODO: Add option for other backbones and outsource the backbone building to another file
        if model_cfg.backbone == "resnet18_imagenet":
            
            # Load pre-trained ResNet18 model
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            
            # Use all layers for our neural network except the final layers (classification head) as the backbone as we are building our own heads
            self.backbone = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4 
            )
            
            # Number of output channels of ResNet18 layer 4 = Number of Input channels for heads
            backbone_out_channels = 512
            
            # Heads: 1x1 Convolutional layers for center, box, and class predictions
            # These turn the feature maps from the backbone into the desired output format. So into these 3 different predictions:
            #       - Center head: 1 output channel (center confidence)
            #       - Box head: 2 output channels (width and height)
            #       - Class head: num_classes output channels (class probabilities)
            self.center_head = nn.Conv2d(backbone_out_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
            self.box_head = nn.Conv2d(backbone_out_channels, out_channels=4, kernel_size=1, stride=1, padding=0)
            self.cls_head = nn.Conv2d(backbone_out_channels, model_cfg.num_classes, kernel_size=1, stride=1, padding=0)

            # Activation function
            # Sigmoid-function converts outputs to probabilities between 0 and 1
            self.sigmoid = nn.Sigmoid()
            
        else:
            raise ValueError(f"Unsupported backbone: {model_cfg.backbone}")
        


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Creates semantics of all pixels on feature map (decides for each pixel if center is located there).
        Creates bounding box dimensions and class probabilities for each pixel on feature map.
        
        Inputs:
        x : torch.Tensor
            Input tensor of shape (B, C, H, W). (Batch, Channel/feature, Height, Width)
        Outputs:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            center_preds shape: (B, 1, H', W') - Center predictions
            box_preds shape: (B, 4, H', W') - Box dimension predictions
            class_preds shape: (B, num_classes, H', W') - Class logits predictions
        """
        feat = self.backbone(x)  # Extract features using the backbone

        center_logits = self.center_head(feat)  # Center predictions
        center_preds = self.sigmoid(center_logits)  # Apply Sigmoid activation
        
        box_logits = self.box_head(feat)  # Box dimension predictions
        box_preds = self.sigmoid(box_logits)  # Apply Sigmoid activation

        cls_preds = self.cls_head(feat)  # Class logits predictions without activation (we take what we get)

        if __debug__:
            self._assert_shapes(x, center_preds, box_preds, cls_preds)

        return center_preds, box_preds, cls_preds


    def _assert_shapes(self, x: torch.Tensor, center_preds: torch.Tensor, box_preds: torch.Tensor, class_preds: torch.Tensor) -> None:
        """
        Asserts that the output shapes of the model are as expected.
        Inputs:
            x : torch.Tensor
            center_preds : torch.Tensor
            box_preds : torch.Tensor
            class_preds : torch.Tensor
        """ 
        if x.ndim != 4 or x.shape[1] != 3:                        #checks for correct input shape
            raise ValueError(f"Input tensor x must have shape (B, 3, H, W), got {x.shape}")
        
        imgsz = int(self.grid_cfg.imgsz)                          #expected input image size
        if x.shape[2] != imgsz or x.shape[3] != imgsz:            #checks for correct input height and width
            raise ValueError(f"Input tensor x must have height and width of {imgsz}, got {x.shape[2]} and {x.shape[3]}")
        
        B = x.shape[0]                                       #batch size
        H = int(self.grid_cfg.H)                             #expected feature map height
        W = int(self.grid_cfg.W)                             #expected feature map width

        if center_preds.shape != (B, 1, H, W):                     #checks for correct shape of center predictions
            raise ValueError(f"Expected center_pred shape (B,1,{H},{W}), got {tuple(center_preds.shape)}")
        
        if box_preds.shape != (B, 4, H, W):                        #checks for correct shape of box predictions
            raise ValueError(f"Expected box_pred shape (B,2,{H},{W}), got {tuple(box_preds.shape)}")
        
        C = int(self.model_cfg.num_classes)
        if class_preds.shape[1:] != (C, H, W):                     #checks for correct number of classes in class predictions
            raise ValueError(f"Expected class_pred shape (B,{C},{H},{W}), got {tuple(class_preds.shape)}")
        
        return