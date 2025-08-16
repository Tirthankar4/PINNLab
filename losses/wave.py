"""
wave.py:
Loss and residual functions for 1D wave equation for PINNs.
"""

import torch
import numpy as np
from .base import diff, mse_loss
from .registry import register_loss

# 1D wave equation: u_tt = c^2 * u_xx

def pde_residue_wave(colloc, net, c):
    net_outputs = net(colloc)
    x = colloc[0]
    t = colloc[1]
    u = net_outputs[:, 0:1]
    u_t = diff(u, t, order=1)
    u_tt = diff(u_t, t, order=1)
    u_x = diff(u, x, order=1)
    u_xx = diff(u_x, x, order=1)

    f = u_tt - c**2 * u_xx
    
    return f

@register_loss('wave_1d')
def wave_loss(net, model, batch, critertion, params):
    #Unpack params
    c = params['c']
    mse_cost_function = critertion
    #unpack batch
    collocation_domain = batch['collocation_domain']
    collocation_IC = batch['collocation_IC']
    #Initial condition points
    x_ic = collocation_IC[0]
    t_ic = collocation_IC[1]

    u_ic_true = batch['u_ic_true']      # u(x,0) = f(x)
    u_t_ic_true = batch['u_t_ic_true']  # u_t(x,0) = g(x)

    if len(collocation_IC) == 3:
        #Parameter embedding: inputs are [x, t, c]
        net_ic_inputs = collocation_IC
        net_dom_inputs = collocation_domain
    else:
        #No parameter embedding: inputs are [x, t]
        net_ic_inputs = [x_ic, t_ic]
        net_dom_inputs = collocation_domain
    
    # Loss for initial displacement
    u_ic_pred = net(net_ic_inputs)
    ic_displacement_loss = mse_cost_function(u_ic_pred, u_ic_true)

    # Loss for initial velocity
    u_t_ic_pred = diff(u_ic_pred, t_ic, order=1)
    ic_velocity_loss = mse_cost_function(u_t_ic_pred, u_t_ic_true)

    # Total initial condition loss
    ic_loss = ic_displacement_loss + ic_velocity_loss
    
    #PDE residuals at collocation points
    f = pde_residue_wave(net_dom_inputs, net, c)
    pde_loss = torch.mean(f ** 2)

    # Periodic boundary condition loss (value and first derivative)
    # For wave equation, we need periodic BCs for both u and u_x
    if len(collocation_IC) == 3:
        # Parameter embedding: c is passed as input
        c_bc = torch.full((1, 1), c, device=x_ic.device, dtype=x_ic.dtype)
        u_bc = model.periodic_BC(net, c_bc, 1, coordinate=1, derivative_order=0, component=0)
        u_x_bc = model.periodic_BC(net, c_bc, 1, coordinate=1, derivative_order=1, component=0)
    else:
        # No parameter embedding: use default periodic BC without parameter
        # Create dummy parameter tensor for compatibility
        dummy_param = torch.full((1, 1), 1.0, device=x_ic.device, dtype=x_ic.dtype)
        u_bc = model.periodic_BC(net, dummy_param, 1, coordinate=1, derivative_order=0, component=0)
        u_x_bc = model.periodic_BC(net, dummy_param, 1, coordinate=1, derivative_order=1, component=0)
    
    bc_loss = u_bc + u_x_bc
    
    return ic_loss, bc_loss, pde_loss

# Helper functions for generating periodic initial conditions

def wave_initial_condition_sine(x, amplitude=1.0, wavenumber=1.0, phase=0.0):
    """
    Generate sine wave initial condition: u(x,0) = A * sin(kx + φ)
    
    Args:
        x: spatial coordinate tensor
        amplitude: wave amplitude A
        wavenumber: wave number k (2π/λ)
        phase: phase shift φ
    
    Returns:
        u_0: initial displacement tensor
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    if x.dim() == 0:
        x = x.unsqueeze(0)
    
    u_0 = amplitude * torch.sin(wavenumber * x + phase)
    
    if u_0.dim() == 1:
        u_0 = u_0.unsqueeze(-1)
    
    return u_0

def wave_initial_condition_gaussian(x, amplitude=1.0, center=0.5, width=0.1):
    """
    Generate Gaussian pulse initial condition: u(x,0) = A * exp(-(x-x₀)²/σ²)
    
    Args:
        x: spatial coordinate tensor
        amplitude: pulse amplitude A
        center: center position x₀
        width: width parameter sigma
    
    Returns:
        u_0: initial displacement tensor
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    if x.dim() == 0:
        x = x.unsqueeze(0)
    
    u_0 = amplitude * torch.exp(-((x - center) ** 2) / (2 * width ** 2))
    
    if u_0.dim() == 1:
        u_0 = u_0.unsqueeze(-1)
    
    return u_0

def wave_initial_velocity_zero(x):
    """
    Generate zero initial velocity: u_t(x,0) = 0
    
    Args:
        x: spatial coordinate tensor
    
    Returns:
        u_t_0: initial velocity tensor (all zeros)
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    if x.dim() == 0:
        x = x.unsqueeze(0)
    
    u_t_0 = torch.zeros_like(x)
    
    if u_t_0.dim() == 1:
        u_t_0 = u_t_0.unsqueeze(-1)
    
    return u_t_0

def wave_initial_velocity_traveling(x, c, amplitude=1.0, wavenumber=1.0, phase=0.0):
    """
    Generate traveling wave initial velocity: u_t(x,0) = -c * u_x(x,0)
    This creates a wave that propagates to the right
    
    Args:
        x: spatial coordinate tensor
        c: wave speed
        amplitude: wave amplitude A
        wavenumber: wave number k
        phase: phase shift φ
    
    Returns:
        u_t_0: initial velocity tensor
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    if x.dim() == 0:
        x = x.unsqueeze(0)
    
    # u_t(x,0) = -c * d/dx[A * sin(kx + φ)] = -c * A * k * cos(kx + φ)
    u_t_0 = -c * amplitude * wavenumber * torch.cos(wavenumber * x + phase)
    
    if u_t_0.dim() == 1:
        u_t_0 = u_t_0.unsqueeze(-1)
    
    return u_t_0
    

