"""
dependency_codes/solver.py
Backwards compatibility wrapper for the new modular solver structure.
This file imports and re-exports functions from the modular solvers for existing code compatibility.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from dependency_codes.config import MODEL_TYPE
import dependency_codes.config as config_module

# Import modular solver functions
from .solvers.hydro import input_taker, req_consts_calc
from .solvers.burgers import burgers_initial_condition
from losses.wave import wave_initial_condition_sine, wave_initial_velocity_zero
from .solvers.base import closure_batched, train_batched_with_progress

# Convenience aliases for backwards compatibility
def wave_initial_condition(x):
    """Backwards compatibility wrapper for wave initial condition"""
    return wave_initial_condition_sine(x, amplitude=1.0, wavenumber=2*np.pi, phase=0.0)

def wave_initial_velocity(x):
    """Backwards compatibility wrapper for wave initial velocity"""  
    return wave_initial_velocity_zero(x)

def process_batch(batch_size, collocation_domain, collocation_IC, alpha, net, model, 
                  mse_cost_function, alpha_idx, alpha_val, **equation_params):
    """
    Process a single batch of data using modular loss function.
    This function routes to the appropriate equation-specific processor based on MODEL_TYPE.
    """
    if MODEL_TYPE == 'hydro':
        from .solvers.hydro import process_batch as hydro_process_batch
        return hydro_process_batch(batch_size, collocation_domain, collocation_IC, alpha, net, model, 
                                   mse_cost_function, alpha_idx, alpha_val, **equation_params)
    
    elif MODEL_TYPE == 'burgers':
        from .solvers.burgers import process_batch as burgers_process_batch
        return burgers_process_batch(batch_size, collocation_domain, collocation_IC, alpha, net, model, 
                                     mse_cost_function, alpha_idx, alpha_val)
    
    elif MODEL_TYPE == 'wave':
        from .solvers.wave import process_batch as wave_process_batch
        return wave_process_batch(batch_size, collocation_domain, collocation_IC, alpha, net, model, 
                                  mse_cost_function, alpha_idx, alpha_val, **equation_params)
    else:
        raise ValueError(f'Unknown MODEL_TYPE: {MODEL_TYPE}')

def closure_batched(model, net, alpha, mse_cost_function, collocation_domain, collocation_IC, 
                    optimizer, lam, jeans, v_1, w_IC, w_BC, w_PDE, batch_size, num_batches):
    """
    Closure function for batched training with stochastic alpha sampling.
    This wrapper uses the base closure_batched function with the appropriate process_batch function.
    """
    # Import the base closure function and use the modular process_batch
    from .solvers.base import closure_batched as base_closure_batched
    
    # Prepare equation-specific parameters for hydro (other equations ignore these)
    equation_params = {}
    if MODEL_TYPE == 'hydro':
        equation_params = {'lam': lam, 'jeans': jeans, 'v_1': v_1}
    
    return base_closure_batched(model, net, alpha, mse_cost_function, collocation_domain, collocation_IC, 
                               optimizer, w_IC, w_BC, w_PDE, batch_size, num_batches,
                               process_batch, **equation_params)

def train_batched_with_progress(training_id, net, model, alpha, collocation_domain, collocation_IC, optimizer, optimizerL, closure, mse_cost_function, 
                  iteration_adam, iterationL, lam, jeans, v_1, batch_size, num_batches, progress_update_fn=None):
    """
    Training function with batched processing and web progress tracking.
    This wrapper uses the base train_batched_with_progress function with the appropriate process_batch function.
    """
    # Import the base training function and use the modular process_batch
    from .solvers.base import train_batched_with_progress as base_train_batched_with_progress
    
    # Prepare equation-specific parameters for hydro (other equations ignore these)
    equation_params = {}
    if MODEL_TYPE == 'hydro':
        equation_params = {'lam': lam, 'jeans': jeans, 'v_1': v_1}
    
    return base_train_batched_with_progress(training_id, net, model, alpha, collocation_domain, collocation_IC, 
                                           optimizer, optimizerL, mse_cost_function, 
                                           iteration_adam, iterationL, batch_size, num_batches, 
                                           process_batch, model_type=MODEL_TYPE, progress_update_fn=progress_update_fn,
                                           **equation_params)