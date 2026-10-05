import torch
from exact.exact_sol import exact_solution_func

def test_pinn_solution(model, boundaries, D: float, N_test: int = 200):
    """
    Evaluate a trained PINN model on a regular grid of (x,t) points.
    Produces:
      - exact solution
      - predicted solution
      - pointwise MSE
    """

    model.eval()

    # ---- 1. Build test grid ----
    t = torch.linspace(boundaries['t_min'], boundaries['t_max'], N_test)
    x = torch.linspace(boundaries['x_min'], boundaries['x_max'], N_test)
    X,T = torch.meshgrid(x,t, indexing="ij")

    coords = torch.stack([ X.reshape(-1),T.reshape(-1)], dim=1)
    coords.requires_grad_(True)

    # ---- 2. Compute predictions ----
    with torch.no_grad():
        pred = model(coords).numpy().reshape(N_test, N_test)

    # ---- 3. Compute exact solution ----
    with torch.no_grad():
        exact = exact_solution_func(coords, diffusion_param=D)
        exact = exact.numpy().reshape(N_test, N_test)

    # ---- 4. Compute pointwise MSE ----
    mse = (pred - exact)**2

    return X.numpy(), T.numpy(), exact, pred, mse
