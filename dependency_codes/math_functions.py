import torch
import numpy as np
from dependency_codes.config import rho_o

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