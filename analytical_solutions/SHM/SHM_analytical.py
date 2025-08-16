"""
SHM_analytical.py
Analytical solutions for 1D Simple Harmonic Oscillator equation for comparison with PINN results
"""

import numpy as np
import torch

def SHM_analytical_solution(t, c, initial_displacement, initial_velocity):
    """
    Analytical solution for 1D Simple Harmonic Oscillator.
    
    ODE: x_tt + c^2 * x = 0
    Solution: x(t) = A*cos(ωt) where ω = √c
    
    Args:
        t: time points (numpy array or torch tensor)
        c: parameter in the ODE (c^2 is the spring constant)
        initial_displacement: initial displacement A (x(0) = A)
        initial_velocity: initial velocity (x_t(0) = 0)
    
    Returns:
        x: analytical solution at time points t
    """
    # Calculate angular frequency
    omega = np.sqrt(c)
    
    # Analytical solution: x(t) = A*cos(ωt)
    # Since initial velocity is 0, we only have the cosine term
    x = initial_displacement * np.cos(omega * t)
    
    return x

def SHM_analytical_solution_exact(t, c, amplitude=1.0, **kwargs):
    """
    Generate exact analytical solution for SHM with specified parameters.
    
    Args:
        t: time coordinate array
        c: spring constant parameter (ω = √c)
        amplitude: initial displacement amplitude
        **kwargs: additional parameters (for consistency with other equations)
    
    Returns:
        x: exact analytical solution
    """
    omega = np.sqrt(c)
    return amplitude * np.cos(omega * t)



def compute_SHM_error(x_pred, x_true, error_type='absolute'):
    """
    Compute error between predicted and true solutions.
    
    Args:
        x_pred: predicted solution
        x_true: true/analytical solution
        error_type: type of error ('absolute', 'relative', 'l2')
    
    Returns:
        error: computed error
    """
    if error_type == 'absolute':
        return x_pred - x_true
    elif error_type == 'relative':
        return (x_pred - x_true) / (np.abs(x_true) + 1e-8)
    elif error_type == 'l2':
        return np.sqrt(np.mean((x_pred - x_true)**2))
    else:
        raise ValueError(f"Unknown error type: {error_type}")
    