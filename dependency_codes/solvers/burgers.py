"""
dependency_codes/solvers/burgers.py
Burgers equation-specific solver functions
"""

import numpy as np
import torch
from losses.burgers import burgers_loss
from .base import closure_batched, train_batched_with_progress

def burgers_initial_condition(x):
    """
    Initial condition for Burgers' equation: u(x,0) = -sin(πx)
    Note: This is only the initial condition, not the full time-dependent solution.
    """
    # Ensure x is a tensor and handle both single values and batches
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    # Handle both single values and batches
    if x.dim() == 0:
        x = x.unsqueeze(0)
    
    # Apply initial condition: u(x,0) = -sin(πx)
    u_0 = -torch.sin(torch.pi * x)
    
    # Add dimension to match network output shape if needed
    if u_0.dim() == 1:
        u_0 = u_0.unsqueeze(-1)
    
    return u_0

def process_batch(batch_size, collocation_domain, collocation_IC, alpha, net, model, 
                  mse_cost_function, alpha_idx, alpha_val, **equation_params):
    """
    Process a single batch of data for Burgers equation
    """
    
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

    # Prepare batch and params for Burgers' - always use parameter embedding
    x_ic = batch_ic[0]
    t_ic = batch_ic[1]
    u_ic_true = burgers_initial_condition(x_ic)
    
    # Always use parameter embedding: nu is passed as input to network (like alpha in hydrodynamics)
    nu_input = torch.full((actual_batch_size, 1), alpha_val.item(), device=x_ic.device, dtype=x_ic.dtype)
    net_ic_inputs = [x_ic, t_ic, nu_input]
    net_dom_inputs = [batch_domain[0], batch_domain[1], nu_input]
    nu_value = alpha_val.item()  # For loss computation
    
    batch_dict = {
        'collocation_domain': net_dom_inputs,
        'collocation_IC': net_ic_inputs,
        'u_ic_true': u_ic_true,
        'batch_size': actual_batch_size
    }
    params_dict = {
        'nu': nu_value
    }
    
    ic_loss, bc_loss, pde_loss = burgers_loss(net, model, batch_dict, mse_cost_function, params_dict)
    
    return ic_loss, bc_loss, pde_loss

def closure_batched_burgers(model, net, alpha, mse_cost_function, collocation_domain, collocation_IC, 
                            optimizer, w_IC, w_BC, w_PDE, batch_size, num_batches):
    """
    Burgers-specific closure function for batched training
    """
    return closure_batched(model, net, alpha, mse_cost_function, collocation_domain, collocation_IC, 
                          optimizer, w_IC, w_BC, w_PDE, batch_size, num_batches,
                          process_batch)

def train_batched_with_progress_burgers(training_id, net, model, alpha, collocation_domain, collocation_IC, 
                                        optimizer, optimizerL, mse_cost_function, 
                                        iteration_adam, iterationL, batch_size, num_batches, 
                                        progress_update_fn=None):
    """
    Burgers-specific training function with batched processing and web progress tracking
    """
    return train_batched_with_progress(training_id, net, model, alpha, collocation_domain, collocation_IC, 
                                       optimizer, optimizerL, mse_cost_function, 
                                       iteration_adam, iterationL, batch_size, num_batches, 
                                       process_batch, model_type='burgers', progress_update_fn=progress_update_fn)
