import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from losses.losses import ASTPN
from losses.hydrodynamic import hydrodynamical_equations_loss, pde_residue_hydro
from losses.burgers import burgers_loss
from model_architecture import PINN
from config import cs, const, G, rho_o, MODEL_TYPE

def input_taker(lam, rho_1, num_of_waves, tmax, N_0, N_b, N_r):
    lam = float(lam)
    rho_1 = float(rho_1)
    num_of_waves = int(num_of_waves)  
    tmax = float(tmax)
    N_0 = int(N_0)
    N_b = int(N_b)
    N_r = int(N_r)
    
    return lam, rho_1, num_of_waves, tmax, N_0, N_b, N_r

def req_consts_calc(lam):

    jeans = np.sqrt(4*np.pi**2*cs**2/(const*G*rho_o))

    if lam > jeans:
        alpha = np.sqrt(const*G*rho_o-cs**2*(2*np.pi/lam)**2)
    else:
        alpha = np.sqrt(cs**2*(2*np.pi/lam)**2 - const*G*rho_o)

    return jeans, alpha

def fun_rho_0(lam, x, alpha):
    """
    Compute initial density with proper tensor sizes
    """
    x_input = x[0]
    alpha = alpha
    
    # Ensure inputs have the same size
    assert x_input.size() == alpha.size(), f"Size mismatch in fun_rho_0: x_input {x_input.size()} vs alpha {alpha.size()}"
    
    rho_0 = rho_o + alpha * torch.cos(2*np.pi*x_input/lam)
    return rho_0

def fun_v_0(lam, jeans, x, v_1):
    """
    Compute initial velocity with proper tensor sizes
    """
    x_input = x[0]
    v_1 = v_1
    
    # Ensure inputs have the same size
    assert x_input.size() == v_1.size(), f"Size mismatch in fun_v_0: x_input {x_input.size()} vs v_1 {v_1.size()}"
    
    if lam > jeans:
        v_0 = - v_1 * torch.sin(2*np.pi*x_input/lam)
    else:
        v_0 = v_1 * torch.cos(2*np.pi*x_input/lam)
    return v_0

def func(x):
    return x[0]*0

from tqdm import tqdm

start = time.time()

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
                  lam, jeans, v_1, mse_cost_function, alpha_idx, alpha_val):
    """
    Process a single batch of data using modular loss function
    """
    
    # Get total number of points
    total_domain_points = collocation_domain[0].size(0)
    total_ic_points = collocation_IC[0].size(0)
    num_alphas = alpha.size(0)
    
    # Use the smaller of batch_size and total_ic_points to ensure we don't exceed available data
    actual_batch_size = min(batch_size, total_ic_points)
    
    # Create batch indices ON GPU
    device = collocation_domain[0].device
    domain_indices = torch.randperm(total_domain_points, device=device)[:actual_batch_size]
    ic_indices = torch.randperm(total_ic_points, device=device)[:actual_batch_size]
    
    # Get batch data and ensure requires_grad is set correctly
    batch_domain = [t[domain_indices].requires_grad_(True) for t in collocation_domain]
    batch_ic = [t[ic_indices].requires_grad_(True) for t in collocation_IC]

    if MODEL_TYPE == 'hydro':
        # Prepare batch and params dicts for modular loss
        batch_dict = {
            'collocation_domain': collocation_domain,
            'collocation_IC': collocation_IC,
            'alpha': alpha,
            'alpha_idx': alpha_idx,
            'alpha_val': alpha_val,
            'batch_size': batch_size
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
    
    elif MODEL_TYPE == 'burgers':
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
        
        # Import burgers_loss here to avoid circular imports
        from losses.burgers import burgers_loss
        ic_loss, bc_loss, pde_loss = burgers_loss(net, model, batch_dict, mse_cost_function, params_dict)
        
        return ic_loss, bc_loss, pde_loss
    
    elif MODEL_TYPE == 'wave':
        # Prepare batch and params for Wave equation - always use parameter embedding
        x_ic = batch_ic[0]
        t_ic = batch_ic[1]
        
        # Wave equation initial condition (sine wave) - match analytical solution
        def wave_initial_condition(x):
            return torch.sin(2 * np.pi * x)
        
        # Initial velocity (zero for standing wave) - match analytical solution
        def wave_initial_velocity(x):
            return torch.zeros_like(x)
        
        u_ic_true = wave_initial_condition(x_ic)
        u_t_ic_true = wave_initial_velocity(x_ic)
        
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
        
        # Import wave_loss here to avoid circular imports
        from losses.wave import wave_loss
        ic_loss, bc_loss, pde_loss = wave_loss(net, model, batch_dict, mse_cost_function, params_dict)
        
        return ic_loss, bc_loss, pde_loss
    else:
        raise ValueError(f'Unknown MODEL_TYPE: {MODEL_TYPE}')

def closure_batched(model, net, alpha, mse_cost_function, collocation_domain, collocation_IC, 
                    optimizer, lam, jeans, v_1, w_IC, w_BC, w_PDE, batch_size, num_batches):
    """
    Closure function for batched training with stochastic alpha sampling
    """
    optimizer.zero_grad()
    
    # Randomly sample alpha indices for this optimization step (on GPU)
    num_alphas = alpha.size(0)
    device = alpha.device
    if num_batches >= num_alphas:
        alpha_indices = torch.arange(num_alphas, device=device)
    else:
        alpha_indices = torch.randperm(num_alphas, device=device)[:num_batches]
    
    total_ic_loss = 0
    total_bc_loss = 0
    total_pde_loss = 0
    
    for batch_idx, alpha_idx in enumerate(alpha_indices):
        alpha_val = alpha[alpha_idx]
        ic_loss, bc_loss, pde_loss = process_batch(
            batch_size, collocation_domain, collocation_IC, alpha,
            net, model, lam, jeans, v_1, mse_cost_function, alpha_idx, alpha_val
        )
        
        total_ic_loss += ic_loss
        total_bc_loss += bc_loss
        total_pde_loss += pde_loss
    
    # Average losses across batches
    ic_loss = total_ic_loss / len(alpha_indices)
    bc_loss = total_bc_loss / len(alpha_indices)
    pde_loss = total_pde_loss / len(alpha_indices)
    
    # Compute total loss with weights
    loss = w_IC * ic_loss + w_BC * bc_loss + w_PDE * pde_loss
    
    # Backward pass
    loss.backward()
    
    # Only return .item() for logging, not for main computation
    return loss, ic_loss, bc_loss, pde_loss

def train_batched_with_progress(training_id, net, model, alpha, collocation_domain, collocation_IC, optimizer, optimizerL, closure, mse_cost_function, 
                  iteration_adam, iterationL, lam, jeans, v_1, batch_size, num_batches, progress_update_fn=None):
    """
    Training function with batched processing and web progress tracking
    """
    print(f"DEBUG: Starting training with {iteration_adam} Adam + {iterationL} L-BFGS iterations")
    
    # Enable debug prints only for the first iteration
    net.debug_print = True
    model.debug_print = True
    
    # Update progress at very beginning
    if progress_update_fn:
        progress_update_fn(training_id, {
            'status': 'training',
            'phase': 'starting',
            'progress': 25,
            'message': 'Initializing training loop...',
            'current_iteration': 0
        })
    
    # Training loop for Adam with progress tracking
    print("Starting Adam optimization...")
    
    # Update progress at start of Adam training
    if progress_update_fn:
        progress_update_fn(training_id, {
            'status': 'training',
            'phase': 'adam_training',
            'progress': 30,
            'message': f'Starting Adam optimization...',
            'current_iteration': 0
        })
    
    for i in range(iteration_adam):
        # Disable debug prints after first iteration
        if i == 1:
            net.debug_print = False
            model.debug_print = False
            
        # Update weights based on iteration and model type
        import config as config_module
        model_type = config_module.MODEL_TYPE
        
        if model_type == 'wave':
            # For wave equation, emphasize initial conditions more
            if i < 200:
                w_IC, w_BC, w_PDE = 10, 1, 1
            elif 200 <= i < 400:
                w_IC, w_BC, w_PDE = 8, 1, 2
            elif 400 <= i < 600:
                w_IC, w_BC, w_PDE = 5, 1, 3
            elif 600 <= i < 800:
                w_IC, w_BC, w_PDE = 3, 1, 5
            else:
                w_IC, w_BC, w_PDE = 2, 1, 8
        else:
            # Original weighting for hydro and burgers
            if i < 200:
                w_IC, w_BC, w_PDE = 1, 1, 2
            elif 200 <= i < 400:
                w_IC, w_BC, w_PDE = 1, 1, 4
            elif 400 <= i < 600:
                w_IC, w_BC, w_PDE = 1, 1, 6
            elif 600 <= i < 800:
                w_IC, w_BC, w_PDE = 1, 1, 8
            else:
                w_IC, w_BC, w_PDE = 1, 1, 10
        
        # Compute loss and update parameters
        loss, ic_loss, bc_loss, pde_loss = closure_batched(
            model, net, alpha, mse_cost_function, collocation_domain,
            collocation_IC, optimizer, lam, jeans, v_1,
            w_IC, w_BC, w_PDE, batch_size, num_batches
        )
        
        # Update progress every 25 iterations (more frequent updates)
        if i % 25 == 0 and progress_update_fn:
            progress_percent = 30 + (i / iteration_adam) * 50  # 30-80% for Adam
            progress_update_fn(training_id, {
                'status': 'training',
                'phase': 'adam_training',
                'progress': int(progress_percent),
                'message': f'Adam training: {i}/{iteration_adam} iterations',
                'current_iteration': i
            })
        
        # Print progress every 100 iterations
        if i % 100 == 0:
            print(f"Training Loss at {i} for Adam in 1D system = {loss:.2e}", flush=True)
        
        optimizer.step()
    
    # Update progress at end of Adam training
    if progress_update_fn:
        progress_update_fn(training_id, {
            'status': 'training',
            'phase': 'adam_completed',
            'progress': 80,
            'message': f'Adam optimization completed. Starting L-BFGS...',
            'current_iteration': iteration_adam
        })
    
    # L-BFGS optimization
    if iterationL > 0:
        print("\nStarting L-BFGS optimization...")
        
        # Update progress to L-BFGS phase
        if progress_update_fn:
            progress_update_fn(training_id, {
                'status': 'training',
                'phase': 'lbfgs_training',
                'progress': 80,
                'message': f'Starting L-BFGS optimization...'
            })
        
        # Create a closure for L-BFGS that captures all required variables
        def lbfgs_closure():
            nonlocal w_IC, w_BC, w_PDE
            # Use the final weights from Adam optimization
            w_IC, w_BC, w_PDE = 1, 1, 10
            
            # Compute loss using closure_batched
            loss, ic_loss, bc_loss, pde_loss = closure_batched(
                model, net, alpha, mse_cost_function, collocation_domain,
                collocation_IC, optimizerL, lam, jeans, v_1,
                w_IC, w_BC, w_PDE, batch_size, num_batches
            )
            
            # Return only the total loss for L-BFGS
            return w_IC * ic_loss + w_BC * bc_loss + w_PDE * pde_loss
        
        # L-BFGS optimization loop with progress tracking
        for i in range(iterationL):
            # Compute loss and update parameters
            loss = optimizerL.step(lbfgs_closure)
            
            # Update progress every 5 iterations (more frequent for L-BFGS)
            if i % 5 == 0 and progress_update_fn:
                progress_percent = 80 + (i / iterationL) * 15  # 80-95% for L-BFGS
                progress_update_fn(training_id, {
                    'status': 'training',
                    'phase': 'lbfgs_training',
                    'progress': int(progress_percent),
                    'message': f'L-BFGS training: {i}/{iterationL} iterations',
                    'current_iteration': iteration_adam + i
                })
            
            # Print progress every 50 iterations
            if i % 50 == 0:
                # Compute detailed losses for printing
                _, ic_loss, bc_loss, pde_loss = closure_batched(
                    model, net, alpha, mse_cost_function, collocation_domain,
                    collocation_IC, optimizerL, lam, jeans, v_1,
                    w_IC, w_BC, w_PDE, batch_size, num_batches
                )
                print(f"Training Loss at {i} for LBGFS in 1D system = {loss:.2e}", flush=True)
    
    # Update progress to finalizing
    if progress_update_fn:
        progress_update_fn(training_id, {
            'status': 'training',
            'phase': 'finalizing',
            'progress': 95,
            'message': 'Finalizing training...'
        })
    
    return net