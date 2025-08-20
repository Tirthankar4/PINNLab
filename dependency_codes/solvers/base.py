import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import threading

# Global stop flags accessible across threads
_global_stop_flags = {}
_global_stop_lock = threading.Lock()

def closure_batched(model, net, alpha, mse_cost_function, collocation_domain, collocation_IC, 
                    optimizer, w_IC, w_BC, w_PDE, batch_size, num_batches,
                    process_batch_func, **equation_params):
    """
    Generic closure function for batched training with stochastic alpha sampling
    
    Args:
        **equation_params: Equation-specific parameters (e.g., lam, jeans, v_1 for hydro)
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
        ic_loss, bc_loss, pde_loss = process_batch_func(
            batch_size, collocation_domain, collocation_IC, alpha,
            net, model, mse_cost_function, alpha_idx, alpha_val, **equation_params
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

def train_batched_with_progress(training_id, net, model, alpha, collocation_domain, collocation_IC, 
                                optimizer, optimizerL, mse_cost_function, 
                                iteration_adam, iterationL, batch_size, num_batches, 
                                process_batch_func, model_type='hydro', progress_update_fn=None, **equation_params):
    """
    Generic training function with batched processing and web progress tracking
    """
    print(f"DEBUG: Starting training with {iteration_adam} Adam + {iterationL} L-BFGS iterations")
    print(f"DEBUG: Training function called with training_id: {training_id}")
    
    # Import here to avoid circular imports
    import app
    
    # Initialize stop flags for this training session
    with _global_stop_lock:
        _global_stop_flags[training_id] = False
    
    with app.stop_flags_lock:
        if training_id not in app.training_stop_flags:
            app.training_stop_flags[training_id] = False
    
    def should_stop_training():
        """Check if training should be stopped"""
        try:
            # First check global flags (more reliable across threads)
            with _global_stop_lock:
                if training_id in _global_stop_flags:
                    global_flag = _global_stop_flags[training_id]
                    if global_flag:
                        print(f"DEBUG: Global stop flag is True for training_id: {training_id}")
                        return True
            
            # Fallback to app-level flags
            import sys
            app_module = sys.modules.get('app')
            if app_module and hasattr(app_module, 'training_stop_flags') and hasattr(app_module, 'stop_flags_lock'):
                with app_module.stop_flags_lock:
                    if training_id in app_module.training_stop_flags:
                        app_flag = app_module.training_stop_flags[training_id]
                        if app_flag:
                            print(f"DEBUG: App stop flag is True for training_id: {training_id}")
                            return True
            
        except Exception as e:
            print(f"DEBUG: Error checking stop flag: {e}")
        return False
    
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
        # Check if training should be stopped
        if should_stop_training():
            print(f"DEBUG: Training stopped by user at Adam iteration {i}")
            if progress_update_fn:
                progress_update_fn(training_id, {
                    'status': 'stopped',
                    'phase': 'stopped',
                    'progress': int(30 + (i / iteration_adam) * 50),
                    'message': f'Training stopped by user at Adam iteration {i}'
                })
            return net
        
        # Check passed - no need for constant debug output
        
        # Disable debug prints after first iteration
        if i == 1:
            net.debug_print = False
            model.debug_print = False
            
        # Update weights based on iteration and model type
        if model_type == "SHM":
            # SHM has no boundary conditions, so w_BC = 0
            if i < int(iteration_adam // 5):
                w_IC, w_BC, w_PDE = 5, 0, 1
            elif int(iteration_adam // 5) <= i < 2 * int(iteration_adam // 5):
                w_IC, w_BC, w_PDE = 4, 0, 1
            elif 2 * int(iteration_adam // 5) <= i < 3 * int(iteration_adam // 5):
                w_IC, w_BC, w_PDE = 3, 0, 1
            elif 3 * int(iteration_adam // 5) <= i < 4 * int(iteration_adam // 5):
                w_IC, w_BC, w_PDE = 2, 0, 1
            else:
                w_IC, w_BC, w_PDE = 1, 0, 1
        elif model_type == "burgers":
            if i < int(iteration_adam // 5):
                w_IC, w_BC, w_PDE = 5, 0, 1
            elif int(iteration_adam // 5) <= i < 2 * int(iteration_adam // 5):
                w_IC, w_BC, w_PDE = 4, 0, 1
            elif 2 * int(iteration_adam // 5) <= i < 3 * int(iteration_adam // 5):
                w_IC, w_BC, w_PDE = 3, 0, 1
            elif 3 * int(iteration_adam // 5) <= i < 4 * int(iteration_adam // 5):
                w_IC, w_BC, w_PDE = 2, 0, 1
            else:
                w_IC, w_BC, w_PDE = 1, 0, 1
        elif model_type == "hydro":
            if i < int(iteration_adam // 5):
                w_IC, w_BC, w_PDE = 5, 1, 1
            elif int(iteration_adam // 5) <= i < 2 * int(iteration_adam // 5):
                w_IC, w_BC, w_PDE = 4, 1, 1
            elif 2 * int(iteration_adam // 5) <= i < 3 * int(iteration_adam // 5):
                w_IC, w_BC, w_PDE = 3, 1, 1
            elif 3 * int(iteration_adam // 5) <= i < 4 * int(iteration_adam // 5):
                w_IC, w_BC, w_PDE = 2, 1, 1
            else:
                w_IC, w_BC, w_PDE = 1, 1, 1
        elif model_type == "wave":
            if i < int(iteration_adam // 5):
                w_IC, w_BC, w_PDE = 10, 1, 1
            elif int(iteration_adam // 5) <= i < 2 * int(iteration_adam // 5):
                w_IC, w_BC, w_PDE = 7, 1, 1
            elif 2 * int(iteration_adam // 5) <= i < 3 * int(iteration_adam // 5):
                w_IC, w_BC, w_PDE = 4, 1, 1
            elif 3 * int(iteration_adam // 5) <= i < 4 * int(iteration_adam // 5):
                w_IC, w_BC, w_PDE = 1, 1, 1
            else:
                w_IC, w_BC, w_PDE = 1, 1, 1
        else:
            if i < int(iteration_adam // 5):
                w_IC, w_BC, w_PDE = 5, 0, 1
            elif int(iteration_adam // 5) <= i < 2 * int(iteration_adam // 5):
                w_IC, w_BC, w_PDE = 4, 0, 1
            elif 2 * int(iteration_adam // 5) <= i < 3 * int(iteration_adam // 5):
                w_IC, w_BC, w_PDE = 3, 0, 1
            elif 3 * int(iteration_adam // 5) <= i < 4 * int(iteration_adam // 5):
                w_IC, w_BC, w_PDE = 2, 0, 1
            else:
                w_IC, w_BC, w_PDE = 1, 0, 1
        
        # Compute loss and update parameters using the equation-specific closure
        loss, ic_loss, bc_loss, pde_loss = closure_batched(
            model, net, alpha, mse_cost_function, collocation_domain,
            collocation_IC, optimizer, w_IC, w_BC, w_PDE, batch_size, num_batches, 
            process_batch_func, **equation_params
        )
        
        # Update progress AND loss info every 25 iterations (synced with progress bar)
        if i % 25 == 0 and progress_update_fn:
            progress_percent = 30 + (i / iteration_adam) * 50  # 30-80% for Adam
            
            # Get loss values for display
            loss_value = loss.item() if hasattr(loss, 'item') else loss
            ic_loss_value = ic_loss.item() if hasattr(ic_loss, 'item') else ic_loss
            bc_loss_value = bc_loss.item() if hasattr(bc_loss, 'item') else bc_loss
            pde_loss_value = pde_loss.item() if hasattr(pde_loss, 'item') else pde_loss
            
            progress_update_fn(training_id, {
                'status': 'training',
                'phase': 'adam_training',
                'progress': int(progress_percent),
                'message': f'Adam training: {i}/{iteration_adam} iterations',
                'current_iteration': i,
                'loss_info': {
                    'total_loss': f"{loss_value:.2e}",
                    'ic_loss': f"{ic_loss_value:.2e}",
                    'bc_loss': f"{bc_loss_value:.2e}",
                    'pde_loss': f"{pde_loss_value:.2e}",
                    'optimizer': 'Adam',
                    'iteration': i,
                    'weights': f"IC:{w_IC}, BC:{w_BC}, PDE:{w_PDE}"
                }
            })
        
        # Print progress every 100 iterations (keep console logging)
        if i % 100 == 0:
            loss_value = loss.item() if hasattr(loss, 'item') else loss
            print(f"Training Loss at {i} for Adam in 1D system = {loss_value:.2e}", flush=True)
        
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
            
            # Check for stop flag inside L-BFGS closure for immediate stopping
            stop_flag_status = should_stop_training()
            if stop_flag_status:
                print(f"DEBUG: Stop detected inside L-BFGS closure, raising exception to force exit")
                # Raise an exception to force L-BFGS to stop
                raise KeyboardInterrupt("Training stopped by user")
            
            # Use the final weights from Adam optimization
            if model_type == "SHM":
                # SHM has no boundary conditions, so w_BC = 0
                w_IC, w_BC, w_PDE = 1, 0, 1
            elif model_type == "hydro":
                w_IC, w_BC, w_PDE = 1, 1, 1
            elif model_type == "burgers":
                w_IC, w_BC, w_PDE = 1, 1, 1
            else:
                w_IC, w_BC, w_PDE = 1, 1, 1
            
            # Compute loss using closure_batched
            loss, ic_loss, bc_loss, pde_loss = closure_batched(
                model, net, alpha, mse_cost_function, collocation_domain,
                collocation_IC, optimizerL, w_IC, w_BC, w_PDE, batch_size, num_batches, 
                process_batch_func, **equation_params
            )
            
            total_loss = w_IC * ic_loss + w_BC * bc_loss + w_PDE * pde_loss
            
            # Return only the total loss for L-BFGS
            return total_loss
        
        # L-BFGS optimization loop with progress tracking
        for i in range(iterationL):
            # Check if training should be stopped BEFORE each L-BFGS step
            if should_stop_training():
                print(f"DEBUG: Training stopped by user BEFORE L-BFGS iteration {i}")
                if progress_update_fn:
                    progress_update_fn(training_id, {
                        'status': 'stopped',
                        'phase': 'stopped',
                        'progress': int(80 + (i / iterationL) * 15),
                        'message': f'Training stopped by user at L-BFGS iteration {i}'
                    })
                return net
            
            # Compute loss and update parameters
            try:
                loss = optimizerL.step(lbfgs_closure)
            except KeyboardInterrupt as e:
                print(f"DEBUG: L-BFGS step {i} interrupted by stop flag: {e}")
                if progress_update_fn:
                    progress_update_fn(training_id, {
                        'status': 'stopped',
                        'phase': 'stopped',
                        'progress': int(80 + (i / iterationL) * 15),
                        'message': f'Training stopped by user during L-BFGS iteration {i}'
                    })
                return net
            except Exception as e:
                print(f"DEBUG: L-BFGS step {i} failed: {e}")
                if should_stop_training():
                    print(f"DEBUG: Stop flag detected during L-BFGS step {i}")
                    return net
                raise e
            
            # Check again AFTER the L-BFGS step
            if should_stop_training():
                print(f"DEBUG: Training stopped by user AFTER L-BFGS iteration {i}")
                if progress_update_fn:
                    progress_update_fn(training_id, {
                        'status': 'stopped',
                        'phase': 'stopped',
                        'progress': int(80 + (i / iterationL) * 15),
                        'message': f'Training stopped by user at L-BFGS iteration {i}'
                    })
                return net
            
            # Update progress AND loss info every 5 iterations (synced with progress bar)
            if i % 5 == 0 and progress_update_fn:
                progress_percent = 80 + (i / iterationL) * 15  # 80-95% for L-BFGS
                
                # Compute detailed losses for display
                _, ic_loss, bc_loss, pde_loss = closure_batched(
                    model, net, alpha, mse_cost_function, collocation_domain,
                    collocation_IC, optimizerL, w_IC, w_BC, w_PDE, batch_size, num_batches, 
                    process_batch_func, **equation_params
                )
                
                loss_value = loss.item() if hasattr(loss, 'item') else loss
                ic_loss_value = ic_loss.item() if hasattr(ic_loss, 'item') else ic_loss
                bc_loss_value = bc_loss.item() if hasattr(bc_loss, 'item') else bc_loss
                pde_loss_value = pde_loss.item() if hasattr(pde_loss, 'item') else pde_loss
                
                total_iteration = iteration_adam + i
                
                progress_update_fn(training_id, {
                    'status': 'training',
                    'phase': 'lbfgs_training',
                    'progress': int(progress_percent),
                    'message': f'L-BFGS training: {i}/{iterationL} iterations',
                    'current_iteration': total_iteration,
                    'loss_info': {
                        'total_loss': f"{loss_value:.2e}",
                        'ic_loss': f"{ic_loss_value:.2e}",
                        'bc_loss': f"{bc_loss_value:.2e}",
                        'pde_loss': f"{pde_loss_value:.2e}",
                        'optimizer': 'L-BFGS',
                        'iteration': total_iteration,  # Continue from Adam iterations
                        'weights': f"IC:{w_IC}, BC:{w_BC}, PDE:{w_PDE}"
                    }
                })
            
            # Print progress every 10 iterations (keep console logging)
            if i % 10 == 0:
                loss_value = loss.item() if hasattr(loss, 'item') else loss
                print(f"Training Loss at {i} for L-BFGS in 1D system = {loss_value:.2e}", flush=True)
    
    # Update progress to finalizing
    if progress_update_fn:
        progress_update_fn(training_id, {
            'status': 'training',
            'phase': 'finalizing',
            'progress': 95,
            'message': 'Finalizing training...'
        })
    
    return net
