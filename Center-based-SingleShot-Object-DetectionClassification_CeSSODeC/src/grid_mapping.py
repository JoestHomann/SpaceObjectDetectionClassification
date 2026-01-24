# grid_mapping.py
# Implements helper functions for target generation in CeSSODeC.
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
# Created:                  23-Jan-2026 11:00:00
# References:
#   None
#
# Revision history:
#   None
#
# Implemented in VSCode 1.108.1
# 2026 in the Applied Machine Learning Course Project

from config import GridConfig
import math


def clamp_to_grid(i: int, j: int, grid: GridConfig) -> tuple[int, int]:
    """
    Forces indices on valid (integer) grid positions.
    Prevents centerpoints from being outside the grid.
    
    i = row index = y in [0...H-1]
    j = column index = x in [0...W-1]
    """
    i = max(0, min(i,grid.H -1))
    j = max(0, min(j,grid.W - 1))
    return i, j



def yolo_norm_to_grid(xc_norm: float, yc_norm: float, grid: GridConfig) -> tuple[int, int]:
    """
    Converts normalized center coordinates (xc_norm, yc_norm) to grid indices (i,j).
    Also clamps the indices to be valid grid positions via clamp_to_grid.
    
    Inputs:
        xc_norm: normalized x center coordinate [0, 1]
        yc_norm: normalized y center coordinate [0, 1]
        grid: GridConfig class/object with grid parameters (imgsz, stride_S, H, W)

    Outputs:
        i: row index on the grid (y direction)
        j: column index on the grid (x direction)

    Mapping:
        gx = (xc_norm * imgsz) / stride_S = xc_norm * W
        gy = (yc_norm * imgsz) / stride_S = yc_norm * H
        j = floor(gx)
        i = floor(gy)
        clamp_to_grid(i, j, grid)
    """

    # Map normalized coords to grid coords (see above)
    gx = xc_norm * float(grid.W)
    gy = yc_norm * float(grid.H)

    # Convert to integer cell indices via floor-function
    j = int(math.floor(gx))
    i = int(math.floor(gy))

    # Clamp to valid bounds by using clamp_to_grid
    return clamp_to_grid(i, j, grid)