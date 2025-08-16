"""
burgers.py:
Loss and residual functions for 1D Burgers' equation for PINNs.
"""

import torch
from .base import diff, mse_loss
from .registry import register_loss

# 1D Burgers' equation: u_t + u u_x = nu u_xx

def pde_residue_burgers(colloc, net, nu):
    net_outputs = net(colloc)
    x = colloc[0]
    t = colloc[1]
    u = net_outputs[:, 0:1]
    u_t = diff(u, t, order=1)
    u_x = diff(u, x, order=1)
    u_xx = diff(u_x, x, order=1)
    # Burgers' equation: u_t + u * u_x = nu * u_xx
    f = u_t + u * u_x - nu * u_xx
    
    return f

@register_loss('burgers_1d')
def burgers_loss(net, model, batch, criterion, params):
    # Unpack params
    nu = params['nu']
    mse_cost_function = criterion
    # Unpack batch
    collocation_domain = batch['collocation_domain']
    collocation_IC = batch['collocation_IC']
    batch_size = batch['batch_size']
    # Initial condition points
    x_ic = collocation_IC[0]
    t_ic = collocation_IC[1]
    u_ic_true = batch['u_ic_true']  # Should be provided in batch
    
    # Check if parameter embedding is used (3 inputs) or not (2 inputs)
    if len(collocation_IC) == 3:
        # Parameter embedding: inputs are [x, t, nu]
        net_ic_inputs = collocation_IC
        net_dom_inputs = collocation_domain
    else:
        # No parameter embedding: inputs are [x, t]
        net_ic_inputs = [x_ic, t_ic]
        net_dom_inputs = collocation_domain
    
    u_ic_pred = net(net_ic_inputs)
    ic_loss = mse_cost_function(u_ic_pred, u_ic_true)
    
    # PDE residuals at collocation points
    f = pde_residue_burgers(net_dom_inputs, net, nu)
    pde_loss = torch.mean(f ** 2)
    
    # Periodic boundary condition loss (value and first derivative)
    # Always use parameter embedding: nu is passed as input (like alpha in hydrodynamics)
    nu_bc = torch.full((1, 1), nu, device=x_ic.device, dtype=x_ic.dtype)
    u_bc = model.periodic_BC(net, nu_bc, 1, coordinate=1, derivative_order=0, component=0)
    u_x_bc = model.periodic_BC(net, nu_bc, 1, coordinate=1, derivative_order=1, component=0)
    
    bc_loss = u_bc + u_x_bc
    return ic_loss, bc_loss, pde_loss 