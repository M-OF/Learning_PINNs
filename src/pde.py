# All PDE studied
import torch
import torch.nn as nn


class HeatEquation1D:
    def __init__(self, D = 0.5):
        self.D = D

    def heat_equation(self, coordinates: torch.Tensor, u: torch.Tensor):
        """Calculate 1D Heat Equation PDE, up until final time T.

        Args:
            coordinates (torch.Tensor): (x,t) time and space coordinates. 
                                        This will be the bulk, not IC or BC
            u (torch.Tensor): Field to compute
        """
        x = coordinates[:, 0:1]
        t = coordinates[:, 1:2]
        
        grad_u = torch.autograd.grad(
            u, coordinates,
            grad_outputs=torch.ones_like(u),
            create_graph=True

        )[0]
        u_x = grad_u[:,0:1]
        u_t = grad_u[:,1:2]

        grad2_u = torch.autograd.grad(
            u_x, coordinates,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True
        )[0]

        u_xx = grad2_u[:, 0:1]

        return u_t - self.D * u_xx
    
    def compute_pde_loss(self, coords, pred):
        heat_eq_pred = self.heat_equation(coords,pred)
        loss = torch.nn.MSELoss()
        return loss(heat_eq_pred, torch.zeros_like(heat_eq_pred))


class HeatPDEConditions:
    def __init__(self, cost_function: None):
        self.cost_function = cost_function
        
        if cost_function is None:
            self.cost_function = nn.MSELoss()


    def heat_bc(self,coordinates):
        # u = 0 for every t if x +- 1
        return torch.zeros(coordinates.shape[0],1)

    def heat_ic(self, coordinates):
        # u = -sin(pi x) if t = 0
        return -torch.sin(torch.pi * coordinates[:,0:1])
    
    def loss_bc(self,coords, pred_bc):
        return self.cost_function(pred_bc, self.heat_bc(coords))

    def loss_ic(self, coords, pred_ic):
        return self.cost_function(pred_ic, self.heat_ic(coords))

    