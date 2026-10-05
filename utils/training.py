# Training functions

from typing import Optional
from src.pde import HeatEquation1D, HeatPDEConditions
from src.generate import generate_data
from exact.exact_sol import compare_to_exact_solution

def step(pinn_net,
         coords,
         ic_coords,
         bc_coords, 
         PDE: HeatEquation1D,
         Conditions: HeatPDEConditions,
         lambda_ic: int,
         lambda_bc: int,
         *args,
         **kwargs):


    # predictions
    pred_pde = pinn_net.forward(coords)
    pred_ic = pinn_net.forward(ic_coords)
    pred_bc = pinn_net.forward(bc_coords)

    # true value, to be substituted with something like FE
    # true_pde = exact_sol(coords)

    # Losses
    pde_loss = PDE.compute_pde_loss(coords, pred_pde)
    ic_loss = Conditions.loss_ic(ic_coords,pred_ic)
    bc_loss = Conditions.loss_bc(bc_coords,pred_bc)

    # Data term, not necessary for this (forward) problem, will be added later 
    #rec_loss = nn.MSELoss(pred_pde, true_pde)

    total_loss = lambda_ic*ic_loss + lambda_bc*bc_loss + 1* pde_loss

    total_loss.backward()

    return {
            "loss": total_loss.detach(),
            "pde_loss": pde_loss.detach(),
            "ic_loss": ic_loss.detach(),
            "bc_loss": bc_loss.detach(),
            "pred_pde": pred_pde.detach(),
            "pred_ic": pred_ic.detach(),
            "pred_bc": pred_bc.detach(),
        }

def train(params):

    history = {}
    run_exact_solution_comparison = {}
    optimizer = params['optim']

    for epoch in range(params['epochs']):
        
        coords, ic_coords, bc_coords = generate_data(boundaries=params['boundaries'],
                                                 pde_batch_size=params['pde_batch_size'],
                                                 ic_batch_size=params['ic_batch_size'],
                                                 bc_batch_size=params['bc_batch_size'])
        optimizer.zero_grad()

        epoch_output = step(
            params['model'],
            coords,
            ic_coords,
            bc_coords, 
            params['PDE'],
            params['Conditions'],
            params['l_ic'],
            params['l_bc'])

        history[epoch+1] = [float(epoch_output['loss']),
                          float(epoch_output['pde_loss']),
                          float(epoch_output['ic_loss']),
                          float(epoch_output['bc_loss'])]

        if epoch%10 == 0:
            print(f"################################ Epoch: {epoch} ################################")
            print("loss:",epoch_output['loss'].item())



        if epoch%50 == 0:
            loss_exact_problem = compare_to_exact_solution(coords,
                                                        ic_coords,
                                                        bc_coords, 
                                                        epoch_output['pred_pde'], 
                                                        epoch_output['pred_ic'],
                                                        epoch_output['pred_bc'],
                                                            params['PDE'].D)
            
            run_exact_solution_comparison[epoch] = loss_exact_problem
            
        optimizer.step()

    model = params["model"]

    return model, history, run_exact_solution_comparison

