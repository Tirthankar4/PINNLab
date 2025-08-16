"""
dependency_codes/solvers/hydro.py
Hydrodynamic-specific solver functions
"""

import numpy as np
import torch
from dependency_codes.config import cs, const, G, rho_o
from losses.hydrodynamic import hydrodynamical_equations_loss
from .base import closure_batched, train_batched_with_progress

def input_taker(lam, rho_1, num_of_waves, tmax, N_0, N_b, N_r):
    """
    Convert input parameters to appropriate types for hydrodynamics
    """
    lam = float(lam)
    rho_1 = float(rho_1)
    num_of_waves = int(num_of_waves)  
    tmax = float(tmax)
    N_0 = int(N_0)
    N_b = int(N_b)
    N_r = int(N_r)
    
    return lam, rho_1, num_of_waves, tmax, N_0, N_b, N_r

def req_consts_calc(lam):
    """
    Calculate required constants for hydrodynamic equations
    """
    jeans = np.sqrt(4*np.pi**2*cs**2/(const*G*rho_o))

    if lam > jeans:
        alpha = np.sqrt(const*G*rho_o-cs**2*(2*np.pi/lam)**2)
    else:
        alpha = np.sqrt(cs**2*(2*np.pi/lam)**2 - const*G*rho_o)

    return jeans, alpha

def process_batch(batch_size, collocation_domain, collocation_IC, alpha, net, model, 
                  mse_cost_function, alpha_idx, alpha_val, **equation_params):
    """
    Process a single batch of data for hydrodynamic equations
    """
    
    # Extract equation-specific parameters
    lam = equation_params.get('lam')
    jeans = equation_params.get('jeans')
    v_1 = equation_params.get('v_1')
    
    # Get total number of points
    total_domain_points = collocation_domain[0].size(0)
    total_ic_points = collocation_IC[0].size(0)
    
    # Use the smaller of batch_size and total_ic_points to ensure we don't exceed available data
    actual_batch_size = min(batch_size, total_ic_points)
    
    # Create batch indices ON GPU
    device = collocation_domain[0].device
    domain_indices = torch.randperm(total_domain_points, device=device)[:actual_batch_size]
    ic_indices = torch.randperm(total_ic_points, device=device)[:actual_batch_size]
    
    # Get batch data and ensure requires_grad is set correctly
    batch_domain = [t[domain_indices].requires_grad_(True) for t in collocation_domain]
    batch_ic = [t[ic_indices].requires_grad_(True) for t in collocation_IC]

    # Prepare batch and params dicts for modular loss
    batch_dict = {
        'collocation_domain': batch_domain,
        'collocation_IC': batch_ic,
        'alpha': alpha,
        'alpha_idx': alpha_idx,
        'alpha_val': alpha_val,
        'batch_size': actual_batch_size
    }
    params_dict = {
        'lam': lam,
        'jeans': jeans,
        'v_1': v_1,
        'mse_cost_function': mse_cost_function
    }
    ic_loss, bc_loss, pde_loss = hydrodynamical_equations_loss(net, model, batch_dict, 
                                                               mse_cost_function, params_dict)
    return ic_loss, bc_loss, pde_loss

def closure_batched_hydro(model, net, alpha, mse_cost_function, collocation_domain, collocation_IC, 
                          optimizer, lam, jeans, v_1, w_IC, w_BC, w_PDE, batch_size, num_batches):
    """
    Hydro-specific closure function for batched training
    """
    return closure_batched(model, net, alpha, mse_cost_function, collocation_domain, collocation_IC, 
                          optimizer, w_IC, w_BC, w_PDE, batch_size, num_batches,
                          process_batch, lam=lam, jeans=jeans, v_1=v_1)

def train_batched_with_progress_hydro(training_id, net, model, alpha, collocation_domain, collocation_IC, 
                                      optimizer, optimizerL, mse_cost_function, 
                                      iteration_adam, iterationL, lam, jeans, v_1, batch_size, num_batches, 
                                      progress_update_fn=None):
    """
    Hydro-specific training function with batched processing and web progress tracking
    """
    return train_batched_with_progress(training_id, net, model, alpha, collocation_domain, collocation_IC, 
                                       optimizer, optimizerL, mse_cost_function, 
                                       iteration_adam, iterationL, batch_size, num_batches, 
                                       process_batch, model_type='hydro', progress_update_fn=progress_update_fn,
                                       lam=lam, jeans=jeans, v_1=v_1)
