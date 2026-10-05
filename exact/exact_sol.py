# Exact solutions when available for comparison

import torch
import torch.nn as nn

def exact_solution_func(coords: torch.Tensor, diffusion_param: float = 0.1) -> torch.Tensor:
    """
    Method to generate the exact solution function.
    """
    pi_fact = float(torch.pi)
    return torch.exp(- (pi_fact**2)*(diffusion_param)*coords[:, 1]) * torch.sin(pi_fact * coords[:,0]) *(-1)


def compare_to_exact_solution(coords, ic, bc, pred, pred_ic, pred_bc, diffusion_param):

    true_ic = exact_solution_func(ic,diffusion_param=diffusion_param).unsqueeze(-1)
    true_bc = exact_solution_func(bc,diffusion_param=diffusion_param).unsqueeze(-1)
    true_pde = exact_solution_func(coords,diffusion_param=diffusion_param).unsqueeze(-1)

    loss = nn.MSELoss()

    loss_ic = loss(pred_ic, true_ic)
    loss_bc = loss(pred_bc, true_bc)
    loss_pde = loss(pred, true_pde)

    total_loss = loss_ic + loss_bc + loss_pde

    return {
        'total_loss' : total_loss.detach(),
        'loss_rexact_ic': loss_ic.detach(),
        'loss_rexact_bc': loss_bc.detach(),
        'loss_rexact_pde': loss_pde.detach(),
    }

