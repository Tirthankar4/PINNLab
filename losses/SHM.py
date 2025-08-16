"""
SHM.py
Loss and residual functions for 1D Simple Harmonic Oscillator.
"""

import torch
import numpy as np
from .base import diff, mse_loss
from .registry import register_loss

# 1D SHM equation: x_tt + c^2 * x = 0

def pde_residue_SHM(colloc, net, c):
    net_outputs = net(colloc)
    t = colloc[0]
    x = net_outputs[:, 0:1]
    x_t = diff(x, t, order = 1)
    x_tt = diff(x_t, t, order = 1)

    f = x_tt + c**2 * x

    return f

@register_loss('SHM_1d')
def SHM_loss(net, model, batch, critertion, params):
    #unpac params
    c = params['c']
    mse_cost_function = critertion
    #unpack batch
    collocation_domain = batch['collocation_domain']
    collocation_IC = batch['collocation_IC']
    
    #Initial condition points
    t_ic = collocation_IC[0]
    
    x_ic_true = batch['x_ic_true']
    x_t_ic_true = batch['x_t_ic_true']

    # Check if parameter embedding is used by looking at the actual tensor dimensions
    # For SHM with parameter embedding: collocation_IC[0] should have shape [batch_size, 2] (t, c)
    # For SHM without parameter embedding: collocation_IC[0] should have shape [batch_size, 1] (t only)
    
    if collocation_IC[0].shape[1] == 2:
        # Parameter embedding: inputs are [t, c] - use as is
        net_ic_inputs = collocation_IC
        net_dom_inputs = collocation_domain
    else:
        # No parameter embedding: inputs are [t] only - need to add parameter
        # This case shouldn't happen with the current setup, but handle it gracefully
        # Ensure net_ic_inputs is a list of tensors, not a single tensor
        net_ic_inputs = [t_ic]
        net_dom_inputs = [collocation_domain[0]]  # Ensure this is also a list

    #Loss for initial displacement
    x_ic_pred = net(net_ic_inputs)
    ic_displacement_loss = mse_cost_function(x_ic_pred, x_ic_true)

    #Loss for initial velocity
    x_t_ic_pred = diff(x_ic_pred, t_ic, order = 1)
    ic_velocity_loss = mse_cost_function(x_t_ic_pred, x_t_ic_true)

    #Total initial condition loss
    ic_loss = ic_displacement_loss + ic_velocity_loss

    #PDE residuals at collocation points
    f = pde_residue_SHM(net_dom_inputs, net, c)
    pde_loss = torch.mean(f ** 2)

    return ic_loss, pde_loss

def SHM_initial_condition(t, amplitude=1.0):
    """
    Generate initial condition for SHM: x(0) = A
    
    Args:
        t: time coordinate tensor (should be zeros for initial condition)
        amplitude: initial displacement amplitude A
    
    Returns:
        x_0: initial displacement tensor
    """
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32)
    
    if t.dim() == 0:
        t = t.unsqueeze(0)
    
    # Initial condition: x(0) = A (constant displacement)
    x_0 = torch.full_like(t, amplitude)
    
    if x_0.dim() == 1:
        x_0 = x_0.unsqueeze(-1)
    
    return x_0

def SHM_initial_velocity(t):
    """
    Generate initial velocity for SHM: x_t(0) = 0
    
    Args:
        t: time coordinate tensor (should be zeros for initial condition)
    
    Returns:
        x_t_0: initial velocity tensor (all zeros)
    """
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32)
    
    if t.dim() == 0:
        t = t.unsqueeze(0)
    
    # Initial velocity: x_t(0) = 0
    x_t_0 = torch.zeros_like(t)
    
    if x_t_0.dim() == 1:
        x_t_0 = x_t_0.unsqueeze(-1)
    
    return x_t_0
