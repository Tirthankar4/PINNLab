"""
dependency_codes/solvers/wave.py
Wave equation-specific solver functions
"""

import numpy as np
import torch
from losses.wave import (
    wave_loss, 
    wave_initial_condition_sine, 
    wave_initial_condition_gaussian,
    wave_initial_velocity_zero
)
from .base import closure_batched, train_batched_with_progress

def process_batch(batch_size, collocation_domain, collocation_IC, alpha, net, model, 
                  mse_cost_function, alpha_idx, alpha_val, initial_condition_type='sine', **kwargs):
    """
    Process a single batch of data for Wave equation
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

    # Prepare batch and params for Wave equation - always use parameter embedding
    x_ic = batch_ic[0]
    t_ic = batch_ic[1]
    
    # Use the proper initial condition functions from losses/wave.py
    # Choose initial condition based on type parameter
    if initial_condition_type == 'sine':
        # Sine wave: u(x,0) = sin(2πx)
        u_ic_true = wave_initial_condition_sine(x_ic, amplitude=1.0, wavenumber=2*np.pi, phase=0.0)
    elif initial_condition_type == 'gaussian':
        # Gaussian pulse: u(x,0) = exp(-(x-0.5)²/0.1²)
        u_ic_true = wave_initial_condition_gaussian(x_ic, amplitude=1.0, center=0.5, width=0.1)
    else:
        # Default to sine wave
        u_ic_true = wave_initial_condition_sine(x_ic, amplitude=1.0, wavenumber=2*np.pi, phase=0.0)
    
    # Initial velocity is always zero for standing wave (can be extended later)
    u_t_ic_true = wave_initial_velocity_zero(x_ic)
    
    # Always use parameter embedding: c (wave speed) is passed as input to network
    c_input = torch.full((actual_batch_size, 1), alpha_val.item(), device=x_ic.device, dtype=x_ic.dtype)
    net_ic_inputs = [x_ic, t_ic, c_input]
    net_dom_inputs = [batch_domain[0], batch_domain[1], c_input]
    c_value = alpha_val.item()  # For loss computation
    
    batch_dict = {
        'collocation_domain': net_dom_inputs,
        'collocation_IC': net_ic_inputs,
        'u_ic_true': u_ic_true,
        'u_t_ic_true': u_t_ic_true,
        'batch_size': actual_batch_size
    }
    params_dict = {
        'c': c_value
    }
    
    ic_loss, bc_loss, pde_loss = wave_loss(net, model, batch_dict, mse_cost_function, params_dict)
    
    return ic_loss, bc_loss, pde_loss

def closure_batched_wave(model, net, alpha, mse_cost_function, collocation_domain, collocation_IC, 
                         optimizer, w_IC, w_BC, w_PDE, batch_size, num_batches):
    """
    Wave-specific closure function for batched training
    """
    return closure_batched(model, net, alpha, mse_cost_function, collocation_domain, collocation_IC, 
                          optimizer, w_IC, w_BC, w_PDE, batch_size, num_batches,
                          process_batch)

def train_batched_with_progress_wave(training_id, net, model, alpha, collocation_domain, collocation_IC, 
                                     optimizer, optimizerL, mse_cost_function, 
                                     iteration_adam, iterationL, batch_size, num_batches, 
                                     progress_update_fn=None, initial_condition_type='sine'):
    """
    Wave-specific training function with batched processing and web progress tracking
    """
    # Create a wrapper function that passes the initial condition type to process_batch
    def wave_process_batch_with_ic(batch_size, collocation_domain, collocation_IC, alpha, net, model, 
                                   mse_cost_function, alpha_idx, alpha_val, **kwargs):
        return process_batch(batch_size, collocation_domain, collocation_IC, alpha, net, model, 
                           mse_cost_function, alpha_idx, alpha_val, 
                           initial_condition_type=initial_condition_type, **kwargs)
    
    return train_batched_with_progress(training_id, net, model, alpha, collocation_domain, collocation_IC, 
                                       optimizer, optimizerL, mse_cost_function, 
                                       iteration_adam, iterationL, batch_size, num_batches, 
                                       wave_process_batch_with_ic, model_type='wave', 
                                       progress_update_fn=progress_update_fn)
