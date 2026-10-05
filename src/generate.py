# Collection of functions for coordinates generation

# Generating data for training the network
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn

from typing import Tuple, List

def generate_data(
        boundaries: dict,
        pde_batch_size: int,
        ic_batch_size: int,
        bc_batch_size: int
                  ):
    """Generate bulk and boundary conditions from min and max
       values of t and x
    Args:
        boundaries (List[float]): list containing x_min, x_max, t_min, t_max in this order
    """
    
    x_min = boundaries['x_min']
    x_max = boundaries['x_max']
    t_min = boundaries['t_min']
    t_max = boundaries['t_max']

    # Bulk; use t_min as offset, define size of interval as difference, 
    # then multiply for as many random elements as in the batch_size
    t_bulk = t_min + (t_max - t_min)*torch.rand(pde_batch_size,1)
    x_bulk = x_min + (x_max - x_min)*torch.rand(pde_batch_size, 1)

    coord_pde = torch.cat((x_bulk,t_bulk), dim=-1)
    coord_pde.requires_grad_(True)

    # IC; t=tmin, x as usual
    coord_ic = torch.cat((
        x_min + (x_max - x_min)*torch.rand(ic_batch_size, 1),
        t_min * torch.ones(ic_batch_size, 1),
    ), dim=-1)

    coord_ic.requires_grad_(True)

    # BC; x=+-1, t as usual
    t_bc =  t_min + (t_max - t_min)*torch.rand(bc_batch_size//2,1)

    coord_bc_pos = torch.cat((
        x_max * torch.ones_like(t_bc),
        t_bc
    ), dim=-1)

    coord_bc_neg = torch.cat((
        x_min * torch.ones_like(t_bc),
        t_bc  
    ), dim=-1)
    
    coord_bc = torch.cat((coord_bc_neg, coord_bc_pos), dim=0)
    coord_bc.requires_grad_(True)

    return coord_pde, coord_ic, coord_bc
    