"""
hydrodynamic.py:
Hydrodynamics-specific loss and residual functions for PINNs.
Contains pde_residue_hydro and hydrodynamical_equations_loss.
"""

import numpy as np
import torch
from dependency_codes.config import cs, const, rho_o
from .registry import register_loss
from .base import diff, mse_loss
from dependency_codes.math_functions import fun_rho_0, fun_v_0, func

@register_loss('hydrodynamical_equations')
def hydrodynamical_equations_loss(net, model, batch, criterion, params):

    # Unpack params
    lam = params['lam']
    jeans = params['jeans']
    v_1 = params['v_1']

    mse_cost_function = criterion
    
    collocation_domain = batch['collocation_domain']
    collocation_IC = batch['collocation_IC']
    alpha = batch['alpha']
    alpha_idx = batch['alpha_idx']
    alpha_val = batch['alpha_val']
    batch_size = batch['batch_size']
    
    device = collocation_domain[0].device
    total_domain_points = collocation_domain[0].size(0)
    total_ic_points = collocation_IC[0].size(0)
    actual_batch_size = min(batch_size, total_ic_points)
    domain_indices = torch.randperm(total_domain_points, device=device)[:actual_batch_size]
    ic_indices = torch.randperm(total_ic_points, device=device)[:actual_batch_size]
    batch_domain = [t[domain_indices] for t in collocation_domain]
    batch_ic = [t[ic_indices] for t in collocation_IC]

    # Network inputs
    x_ic, t_ic = batch_ic[0], batch_ic[1]
    alpha_input = torch.full((actual_batch_size, 1), alpha_val.item(), device=x_ic.device, dtype=x_ic.dtype)
    v1_input = torch.full((actual_batch_size, 1), v_1[alpha_idx].item(), device=x_ic.device, dtype=x_ic.dtype)
    net_ic_inputs = [x_ic, t_ic, alpha_input]
    net_dom_inputs = [batch_domain[0], batch_domain[1], alpha_input]

    # Forward pass
    net_ic_output = net(net_ic_inputs)

    rho_0 = fun_rho_0(lam, [x_ic, t_ic], alpha_input)
    vx_0 = fun_v_0(lam, jeans, [x_ic, t_ic], v1_input)
    rho_ic_out = net_ic_output[:,0:1]
    vx_ic_out = net_ic_output[:,1:2]
    mse_rho_ic = mse_cost_function(rho_ic_out, rho_0)
    mse_vx_ic = mse_cost_function(vx_ic_out, vx_0)

    # PDE residuals
    residuals = pde_residue_hydro(net_dom_inputs, net, dimension=1)
    rho_r, vx_r, phi_r = residuals
    mse_rho = torch.mean(rho_r ** 2)
    mse_velx = torch.mean(vx_r ** 2)
    mse_phi = torch.mean(phi_r ** 2)

    # Boundary conditions
    alpha_bc = torch.tensor([[alpha_val]], device=x_ic.device, dtype=x_ic.dtype)
    rhox_b = model.periodic_BC(net, alpha_bc, 1, coordinate=1, derivative_order=0, component=0)
    vx_xb = model.periodic_BC(net, alpha_bc, 1, coordinate=1, derivative_order=0, component=1)
    phi_xb = model.periodic_BC(net, alpha_bc, 1, coordinate=1, derivative_order=0, component=2)
    phi_xx_b = model.periodic_BC(net, alpha_bc, 1, coordinate=1, derivative_order=1, component=2)

    # Combine losses
    ic_loss = mse_rho_ic + mse_vx_ic
    bc_loss = rhox_b + vx_xb + phi_xb + phi_xx_b
    pde_loss = mse_rho + mse_velx + mse_phi
    return ic_loss, bc_loss, pde_loss

def pde_residue_hydro(colloc, net, dimension=1):
    net_outputs = net(colloc)
    x = colloc[0]
    if dimension == 1:
        t = colloc[1]
        rho, vx = net_outputs[:,0:1], net_outputs[:,1:2]
        phi = net_outputs[:,2:3]
        rho_t = diff(rho, t, order=1)
        rho_x = diff(rho, x, order=1)
        vx_t = diff(vx, t, order=1)
        vx_x = diff(vx, x, order=1)
        phi_x = diff(phi, x, order=1)
        phi_x_x = diff(phi, x, order=2)

        rho_r = rho_t + vx * rho_x + rho * vx_x
        vx_r = rho * vx_t + rho * (vx * vx_x) + cs * cs * rho_x + rho * phi_x
        phi_r = phi_x_x - const * (rho - rho_o)
        return rho_r, vx_r, phi_r
    elif dimension == 2:
        y = colloc[1]
        t = colloc[2]
        rho, vx = net_outputs[:,0:1], net_outputs[:,1:2]
        vy = net_outputs[:,2:3]
        phi = net_outputs[:,3:4]
        rho_t = diff(rho, t, order=1)
        rho_x = diff(rho, x, order=1)
        rho_y = diff(rho, y, order=1)
        vx_t = diff(vx, t, order=1)
        vy_t = diff(vy, t, order=1)
        vx_x = diff(vx, x, order=1)
        vx_y = diff(vx, y, order=1)
        vy_x = diff(vy, x, order=1)
        vy_y = diff(vy, y, order=1)
        phi_x = diff(phi, x, order=1)
        phi_x_x = diff(phi, x, order=2)
        phi_y = diff(phi, y, order=1)
        phi_y_y = diff(phi, y, order=2)

        rho_r = rho_t + vx * rho_x + vy * rho_y + rho * vx_x + rho * vy_y
        vx_r = rho * vx_t + rho * (vx * vx_x + vy * vx_y) + cs * cs * rho_x + rho * phi_x
        vy_r = rho * vy_t + rho * (vy * vy_y + vx * vy_x) + cs * cs * rho_y + rho * phi_y
        phi_r = phi_x_x + phi_y_y - const * (rho - rho_o)
        return rho_r, vx_r, vy_r, phi_r
    elif dimension == 3:
        y = colloc[1]
        z = colloc[2]
        t = colloc[3]
        rho, vx = net_outputs[:,0:1], net_outputs[:,1:2]
        vy = net_outputs[:,2:3]
        vz = net_outputs[:,3:4]
        phi = net_outputs[:,4:5]
        rho_t = diff(rho, t, order=1)
        rho_x = diff(rho, x, order=1)
        rho_y = diff(rho, y, order=1)
        rho_z = diff(rho, z, order=1)
        vx_t = diff(vx, t, order=1)
        vy_t = diff(vy, t, order=1)
        vz_t = diff(vz, t, order=1)
        vx_x = diff(vx, x, order=1)
        vy_x = diff(vy, x, order=1)
        vz_x = diff(vz, x, order=1)
        vx_y = diff(vx, y, order=1)
        vy_y = diff(vy, y, order=1)
        vz_y = diff(vz, y, order=1)
        vx_z = diff(vx, z, order=1)
        vy_z = diff(vy, z, order=1)
        vz_z = diff(vz, z, order=1)
        phi_x = diff(phi, x, order=1)
        phi_x_x = diff(phi, x, order=2)
        phi_y = diff(phi, y, order=1)
        phi_y_y = diff(phi, y, order=2)
        phi_z = diff(phi, z, order=1)
        phi_z_z = diff(phi, z, order=2)

        rho_r = (rho_t + vx * rho_x + rho * vx_x + vy * rho_y + rho * vy_y + vz * rho_z + rho * vz_z)
        vx_r = rho * vx_t + rho * (vx * vx_x + vy * vx_y + vz * vx_z) + cs * cs * rho_x + rho * phi_x
        vy_r = rho * vy_t + rho * (vy * vy_y + vx * vy_x + vz * vy_z) + cs * cs * rho_y + rho * phi_y
        vz_r = rho * vz_t + rho * (vz * vz_z + vx * vz_x + vy * vz_y) + cs * cs * rho_z + rho * phi_z
        phi_r = phi_x_x + phi_y_y + phi_z_z - const * (rho - rho_o)
        return rho_r, vx_r, vy_r, vz_r, phi_r
    else:
        raise NotImplementedError('Only 1D, 2D, and 3D hydrodynamics are supported in pde_residue_hydro.')