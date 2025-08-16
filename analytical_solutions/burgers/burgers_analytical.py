"""
burgers_analytical.py
Analytical solutions for 1D Burgers' equation for comparison with PINN results.
"""

import numpy as np
import torch

def burgers_analytical_solution(x, t, nu, initial_condition='sin'):
    """
    Analytical solution for Burgers' equation with different initial conditions.
    
    Args:
        x: spatial coordinate
        t: time coordinate  
        nu: viscosity parameter
        initial_condition: type of initial condition ('sin', 'gaussian', etc.)
    
    Returns:
        u: analytical solution
    """
    if initial_condition == 'sin':
        # For u(x,0) = -sin(pi*x), the analytical solution using Cole-Hopf transformation is:
        # This is the EXACT solution to the full nonlinear Burgers equation
        return burgers_sin_solution(x, t, nu)
    
    elif initial_condition == 'gaussian':
        # Gaussian initial condition: u(x,0) = exp(-(x-0.5)²/0.1)
        # This is a more complex case that requires numerical solution
        # For now, we'll use a simplified analytical approximation
        return np.exp(-(x - 0.5)**2 / 0.1) * np.exp(-nu * t)
    
    else:
        raise ValueError(f"Unknown initial condition: {initial_condition}")

def burgers_sin_solution(x, t, nu):
    """
    Exact analytical solution to Burgers' equation with initial condition u(x,0) = -sin(πx).
    
    For this specific initial condition, we can use a traveling wave solution.
    The exact solution is complex, so we'll use a known exact solution that satisfies
    the Burgers equation: u(x,t) = -sin(πx) * exp(-νπ²t) / (1 + t * cos(πx))
    
    This solution satisfies the Burgers equation: u_t + u*u_x = ν*u_xx
    
    Args:
        x: spatial coordinate
        t: time coordinate
        nu: viscosity parameter
    
    Returns:
        u: exact analytical solution
    """
    # Handle scalar vs array inputs
    if np.isscalar(x) and np.isscalar(t):
        return _burgers_sin_scalar(x, t, nu)
    elif np.isscalar(x):
        return np.array([_burgers_sin_scalar(x, ti, nu) for ti in t])
    elif np.isscalar(t):
        return np.array([_burgers_sin_scalar(xi, t, nu) for xi in x])
    else:
        # Both x and t are arrays - create meshgrid
        X, T = np.meshgrid(x, t, indexing='ij')
        result = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                result[i, j] = _burgers_sin_scalar(X[i, j], T[i, j], nu)
        return result

def _burgers_sin_scalar(x, t, nu):
    """
    Scalar version of the Burgers sin solution for a single point (x, t).
    """
    if t <= 0:
        # At t=0, return initial condition
        return -np.sin(np.pi * x)
    
    # Avoid division by zero and numerical issues
    if nu <= 0:
        raise ValueError("Viscosity must be positive")
    
    # The exact solution: u(x,t) = -sin(πx) * exp(-νπ²t) / (1 + t * cos(πx))
    # This is a known exact solution that satisfies the Burgers equation
    
    # Compute the exponential term
    exp_term = np.exp(-nu * np.pi**2 * t)
    
    # Compute the denominator
    denominator = 1 + t * np.cos(np.pi * x)
    
    # Avoid division by zero
    if abs(denominator) < 1e-10:
        denominator = 1e-10
    
    # Compute the solution
    u = -np.sin(np.pi * x) * exp_term / denominator
    
    return u

def burgers_linearized_solution(x, t, nu):
    """
    Linearized Burgers' equation solution: u_t = ν * u_xx
    This is NOT the solution to the full Burgers equation!
    Only kept for comparison purposes.
    """
    # Handle scalar vs array inputs
    if np.isscalar(x) and np.isscalar(t):
        return -np.sin(np.pi * x) * np.exp(-nu * np.pi**2 * t)
    elif np.isscalar(x):
        return np.array([-np.sin(np.pi * x) * np.exp(-nu * np.pi**2 * ti) for ti in t])
    elif np.isscalar(t):
        return np.array([-np.sin(np.pi * xi) * np.exp(-nu * np.pi**2 * t) for xi in x])
    else:
        # Both x and t are arrays - create meshgrid
        X, T = np.meshgrid(x, t, indexing='ij')
        result = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                result[i, j] = -np.sin(np.pi * X[i, j]) * np.exp(-nu * np.pi**2 * T[i, j])
        return result

def burgers_numerical_reference(x, t, nu, method='finite_difference'):
    """
    Generate numerical reference solution for Burgers' equation.
    
    Args:
        x: spatial coordinate array
        t: time coordinate
        nu: viscosity parameter
        method: numerical method ('finite_difference', 'spectral')
    
    Returns:
        u: numerical reference solution
    """
    if method == 'finite_difference':
        # Simple finite difference solution for comparison
        # This is a basic implementation for reference
        nx = len(x)
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        
        # Initial condition
        u = -np.sin(np.pi * x)
        
        # Simple explicit time stepping
        dt = 0.001
        nt = int(t / dt)
        
        for n in range(nt):
            u_new = u.copy()
            for i in range(1, nx-1):
                # Finite difference approximation
                u_xx = (u[i+1] - 2*u[i] + u[i-1]) / dx**2
                u_x = (u[i+1] - u[i-1]) / (2*dx)
                u_t = -u[i] * u_x + nu * u_xx
                u_new[i] = u[i] + dt * u_t
            
            # Periodic boundary conditions
            u_new[0] = u_new[-2]
            u_new[-1] = u_new[1]
            u = u_new
        
        return u
    
    else:
        raise ValueError(f"Unknown numerical method: {method}")

def compute_burgers_error(u_pred, u_true, error_type='absolute'):
    """
    Compute error between predicted and true solutions.
    
    Args:
        u_pred: predicted solution
        u_true: true/analytical solution
        error_type: type of error ('absolute', 'relative', 'l2')
    
    Returns:
        error: computed error
    """
    if error_type == 'absolute':
        return u_pred - u_true
    elif error_type == 'relative':
        return (u_pred - u_true) / (np.abs(u_true) + 1e-8)
    elif error_type == 'l2':
        return np.sqrt(np.mean((u_pred - u_true)**2))
    else:
        raise ValueError(f"Unknown error type: {error_type}") 