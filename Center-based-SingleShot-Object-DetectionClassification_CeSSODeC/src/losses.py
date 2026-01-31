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
#   - Removed initial center loss implementation via positive-only negative 
#     log likelihood at gt-pixel with a BCEwithLogits implementation (26-Jan-2026, J. Homann)
#   - Rewrote SingleObjectLoss to sample predicted heatmap at gt centerpoint location (28-Jan-2026, J. Homann, C. Kern)
#
# Implemented in VSCode 1.108.1
# 2026 in the Applied Machine Learning Course Project

import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian_heatmap(heatmap, y_center, x_center, sigma):
    """
    heatmap with gaussian "heat" distribution around the center

    Input:
        heatmap: (H,W)
        y_center: int grid loc y
        x_center: int grid loc x
        sigma: float        
    """
    H, W = heatmap.shape    # Getting heatmap shape
    radius =int(3 * sigma)  # Defines gaussian radius
    
    # Defining the square area around the center where the gaussian will be applied
    y0 = max(0, y_center - radius)          # clamping, and corner values at gaussian radius 
    y1 = min(H, y_center + radius + 1)      
    x0 = max(0, x_center - radius)
    x1 = min(W, x_center + radius +1)

    # Creating meshgrid for the defined area
    yy, xx = torch.meshgrid(
        torch.arange(y0, y1, device=heatmap.device),
        torch.arange(x0, x1, device=heatmap.device),
        indexing = "ij"
    )
    
    # Calculating the Gaussian values
    gaussian = torch.exp(
        -((yy-y_center)**2 + (xx-x_center)**2)/(2*sigma**2)
    )

    # Updating the heatmap with the maximum values between existing and new gaussian values
    heatmap[y0:y1, x0:x1] = torch.maximum(
        heatmap[y0:y1, x0:x1],
        gaussian
    )

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
    
    def __init__(self, eps=1e-6, box_weight: float = 5.0, center_weight: float = 3.0) -> None:
        super(SingleObjectLoss, self).__init__()
        self.eps = eps
        self.box_weight = box_weight                                    # box weight to balance the losses to more exact boxes (higher value = more exact boxes) TODO: tune
        self.center_weight = center_weight                              # center weight to balance the losses (higher value = more exact centers) TODO: tune
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(
            self,
            center_preds: torch.Tensor, 
            box_preds: torch.Tensor, 
            class_preds: torch.Tensor, 
            gridIndices_gt: torch.Tensor, 
            bbox_gt_norm: torch.Tensor, 
            cls_gt: torch.Tensor,
            gaussHm_sigma: float = 2.0,
            BCE_scale: float = 1.0
            ) -> dict[str, torch.Tensor]:
        
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
        # Get batch size, height, and width from center_preds
        B, _, H, W = center_preds.shape     # Dont need channel dimension for center_preds, need batch size, height and width
        
        # Target Heatmap
        # Outputs 1 at ground truth center locations, 0 elsewhere
        center_target = torch.zeros((B, 1, H, W),               # Fill matrix/tensor of size (B,1,H,W) with zeros
                                     device=device,
                                     dtype=center_preds.dtype) 
        center_target = torch.zeros((B, 1, H, W), device = device, dtype=center_preds.dtype)        # Initialize target heatmap with zeros

        standardDeviation = float(gaussHm_sigma)  # Standard deviation for Gaussian heatmap
        # Create Gaussian heatmaps around ground truth center locations
        for batch_index in range(B):
            gaussian_heatmap(center_target[batch_index, 0],
                             y_center = int(i[batch_index].item()),  # y_center = grid row ground truth index, item() converts single-value tensor to int
                             x_center = int(j[batch_index].item()),  # x_center = grid column ground truth index
                             sigma=standardDeviation)                # sigma = standard deviation of the Gaussian
    

        # Positive counter weight to balance out the many negative samples
        # Makes the ground truth pixel (where object center is) as important as all not-gt pixels together
        k = float(BCE_scale)  # Scaling factor to reduce the weight of positive samples to increase stability (TODO: Make this a tuneable hyperparameter)

        # Center loss calculation using binary cross-entropy with logits ("raw" model output)
        Loss_center_map = F.binary_cross_entropy_with_logits(
            center_preds,
            center_target,
            reduction='none'                                    # Ouput the mean loss over the batch
        )

        alpha = k  # Weighting factor for positive samples (TODO: Make this a tuneable hyperparameter)
        beta = 1.0  # Weighting factor for negative samples (TODO: Make this a tuneable hyperparameter)
        weight = alpha * center_target + beta * (1.0 - center_target)       # Weight map for balancing positive and negative samples
        Loss_center = (Loss_center_map * weight).sum() / (weight.sum()+1e-6)  # Weighted center loss

        
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
        Loss_total = self.center_weight * Loss_center + self.box_weight * Loss_box + Loss_class # total loss with box weight

        return {
            'Loss_total': Loss_total,
            'Loss_center': Loss_center,
            'Loss_box': Loss_box,
            'Loss_class': Loss_class
        }
    