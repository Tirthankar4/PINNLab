"""
dependency_codes/solvers/SHM.py
Simple Harmonic Motion (SHM) equation-specific solver functions
"""

import numpy as np
import torch
from losses.SHM import SHM_loss, SHM_initial_condition, SHM_initial_velocity
from .base import closure_batched, train_batched_with_progress

def process_batch(batch_size, collocation_domain, collocation_IC, alpha, net, model, 
                  mse_cost_function, alpha_idx, alpha_val, **equation_params):
    """
    Process a single batch of data for SHM equation
    """
    
    # For SHM, we only need time coordinates (no spatial dimension)
    # collocation_domain should be [t_coords] and collocation_IC should be [t_coords]
    
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
    # For SHM: collocation_domain[0] = time coordinates, collocation_IC[0] = time coordinates
    batch_domain = [collocation_domain[0][domain_indices].requires_grad_(True)]
    batch_ic = [collocation_IC[0][ic_indices].requires_grad_(True)]

    # Prepare batch and params for SHM - always use parameter embedding
    t_ic = batch_ic[0]
    
    # Initial conditions
    x_ic_true = SHM_initial_condition(t_ic, amplitude=1.0)
    x_t_ic_true = SHM_initial_velocity(t_ic)
    
    # Always use parameter embedding: c is passed as input to network
    # Handle both tensor and float inputs for alpha_val
    if hasattr(alpha_val, 'item'):
        c_value = alpha_val.item()
    else:
        c_value = float(alpha_val)
    
    c_input = torch.full((actual_batch_size, 1), c_value, device=t_ic.device, dtype=t_ic.dtype)
    
    # Create single tensors with both time and parameter features
    # This matches what the loss function expects: [tensor([t, c])]
    net_ic_inputs = [torch.cat([t_ic, c_input], dim=1)]
    net_dom_inputs = [torch.cat([batch_domain[0], c_input], dim=1)]
    
    batch_dict = {
        'collocation_domain': net_dom_inputs,
        'collocation_IC': net_ic_inputs,
        'x_ic_true': x_ic_true,
        'x_t_ic_true': x_t_ic_true,
        'batch_size': actual_batch_size
    }
    params_dict = {
        'c': c_value
    }
    
    ic_loss, pde_loss = SHM_loss(net, model, batch_dict, mse_cost_function, params_dict)
    
    # SHM doesn't have boundary conditions (only initial conditions)
    bc_loss = torch.tensor(0.0, device=ic_loss.device)
    
    return ic_loss, bc_loss, pde_loss

def closure_batched_SHM(model, net, alpha, mse_cost_function, collocation_domain, collocation_IC, 
                        optimizer, w_IC, w_BC, w_PDE, batch_size, num_batches):
    """
    SHM-specific closure function for batched training
    """
    return closure_batched(model, net, alpha, mse_cost_function, collocation_domain, collocation_IC, 
                          optimizer, w_IC, w_BC, w_PDE, batch_size, num_batches,
                          process_batch)

def train_batched_with_progress_SHM(training_id, net, model, alpha, collocation_domain, collocation_IC, 
                                    optimizer, optimizerL, mse_cost_function, 
                                    iteration_adam, iterationL, batch_size, num_batches, 
                                    progress_update_fn=None):
    """
    SHM-specific training function with batched processing and web progress tracking
    """
    return train_batched_with_progress(training_id, net, model, alpha, collocation_domain, collocation_IC, 
                                       optimizer, optimizerL, mse_cost_function, 
                                       iteration_adam, iterationL, batch_size, num_batches, 
                                       process_batch, model_type='SHM', progress_update_fn=progress_update_fn)
