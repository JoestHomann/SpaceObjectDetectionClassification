# losses.py
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
# Created:                  24-Jan-2026 14:45:00
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
import torch.nn.functional as F

class SingleObjectLoss(nn.Module):
    """
    Calculates the loss for the centerpoint, the size of the bounding box, and the class label.
    Samples the predicted heatmap at the ground truth centerpoint location to compute the centerpoint loss.
    Computes the size loss using Smooth L1 Loss between predicted and ground truth sizes.
    Computes the classification loss using Cross Entropy Loss between predicted and ground truth class labels.
    
    Losses: sampled at the gt cell (i,j) per image b in B(atch)
        Loss_center = -log(center_preds[b,0,i,j]+ eps)
            Loss_center is the Centerpoint Loss
            -log is the Negative Log Likelihood (for p = 1 NLL is 0, so no loss, if p is small loss is high)
            eps is a small constant to avoid log(0)
        Loss_box = SmoothL1(box_preds[b,:,i,j], box_gt[b,:,i,j])
            Loss_box is the Bounding Box Size Loss
            SmoothL1 is the Smooth L1 Loss function
        Loss_class = CrossEntropy(class_preds[b,:,i,j], class_gt[b,i,j])
            Loss_class is the Classification Loss
            CrossEntropy is the Cross Entropy Loss function
    
    Inputs:
        - center_preds: Predicted heatmap for centerpoints (B, 1, H, W)
        - box_preds: Predicted bounding box sizes (B, 2, H, W)
        - class_preds: Predicted class scores (B, num_classes, H, W)
        - center_gt: Ground truth centerpoint locations (B, 1, H, W)
        - box_gt: Ground truth bounding box sizes (B, 2, H, W)
        - class_gt: Ground truth class labels (B, H, W)
    Outputs:
        - total_loss: Combined loss from centerpoint, bounding box size, and classification losses
    """
    
    def __init__(self, eps=1e-6, box_weight: float = 5.0) -> None:
        super(SingleObjectLoss, self).__init__()
        self.eps = eps
        self.box_weight = box_weight                                    # box weight to balance the losses to more exact boxes (higher value = more exact boxes) TODO: tune
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(
            self,
            center_preds: torch.Tensor, 
            box_preds: torch.Tensor, 
            class_preds: torch.Tensor, 
            gridIndices_gt: torch.Tensor, 
            bbox_gt_norm: torch.Tensor, 
            cls_gt: torch.Tensor) -> dict[str, torch.Tensor]:
        
        """
        Normalizes Batch size shapes (i.e. only one sample in batch) to avoid shape errors.
        Validates that the batch sizes of all inputs match.
        """
        if gridIndices_gt.ndim == 1:
            gridIndices_gt = gridIndices_gt.unsqueeze(0)  #(1,2)
        if bbox_gt_norm.ndim == 1:
            bbox_gt_norm = bbox_gt_norm.unsqueeze(0)    #(1,4)
        if cls_gt.ndim == 0:
            cls_gt = cls_gt.unsqueeze(0)                #(1,)
        
        B = center_preds.shape[0]  # Batch size
        device = center_preds.device                     # Device (CPU or GPU) (targets and model has to be on the same device
        
        if gridIndices_gt.shape[0] != B:
            raise ValueError(f"Batch size of gridIndices_gt ({gridIndices_gt.shape[0]}) does not match batch size of predictions ({B}).")
        if bbox_gt_norm.shape[0] != B:
            raise ValueError(f"Batch size of bbox_gt_norm ({bbox_gt_norm.shape[0]}) does not match batch size of predictions ({B}).")
        if cls_gt.shape[0] != B:
            raise ValueError(f"Batch size of cls_gt ({cls_gt.shape[0]}) does not match batch size of predictions ({B}).")
        
        gridIndices_gt = gridIndices_gt.to(device=device, dtype=torch.int64) # forces dtype and device
        bbox_gt_norm = bbox_gt_norm.to(device=device, dtype=torch.float32)
        cls_gt = cls_gt.to(device=device, dtype=torch.int64)

        i = gridIndices_gt[:, 0]   # row (y)
        j = gridIndices_gt[:, 1]   # col (x)
        b = torch.arange(B, device=device)  # batch indexing

        """
        center point loss
        """
        center_at_gt = center_preds[b, 0, i, j]  # (B,)
        Loss_center = (-torch.log(center_at_gt + self.eps)).mean()  # mean over batch
        
        """
        bounding box size loss
        linear loss with smooting at 0
        """
        box_at_gt = box_preds[b, :, i, j]  # (B, 4)
        Loss_box_raw = self.smooth_l1_loss(box_at_gt, bbox_gt_norm)  # (B,4)
        Loss_box = Loss_box_raw.mean()  # Skalar

        """
        classification loss
        cross entropy loss punishes wrong class predictions more than correct ones
            - class_preds has logits as output (not interpretable probabilities)
            - crossentropy takes logits and transforms them internally to probabilities with softmax
            - then computes negative log likelihood
        """
        cls_logits_at_gt = class_preds[b, :, i, j]  # (B, num_classes)
        Loss_class_raw = self.cross_entropy_loss(cls_logits_at_gt, cls_gt)  # (B,)
        Loss_class = Loss_class_raw.mean()  # Skalar

        """
        total loss
        added losses together with box weight
        """
        total_loss = Loss_center + self.box_weight * Loss_box + Loss_class # total loss with box weight

        return {
            'total_loss': total_loss,
            'Loss_center': Loss_center,
            'Loss_box': Loss_box,
            'Loss_class': Loss_class
        }
    