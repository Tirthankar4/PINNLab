"""
wave_analytical.py
Analytical solutions for 1D wave equation for comparison with PINN results.
"""

import numpy as np
import torch

def wave_analytical_solution_dalembert(x, t, c, initial_displacement, initial_velocity=None):
    """
    d'Alembert solution for the 1D wave equation: u_tt = c^2 * u_xx
    
    u(x,t) = (1/2)[f(x+ct) + f(x-ct)] + (1/(2c))∫_{x-ct}^{x+ct} g(ξ) dξ
    
    Args:
        x: spatial coordinate
        t: time coordinate  
        c: wave speed
        initial_displacement: function f(x) = u(x,0)
        initial_velocity: function g(x) = u_t(x,0) (optional, defaults to zero)
    
    Returns:
        u: analytical solution
    """
    if initial_velocity is None:
        # If no initial velocity provided, assume g(x) = 0
        # Then u(x,t) = (1/2)[f(x+ct) + f(x-ct)]
        return 0.5 * (initial_displacement(x + c * t) + initial_displacement(x - c * t))
    else:
        # For non-zero initial velocity, we need to compute the integral
        # This is more complex and requires numerical integration
        # For now, we'll use a simplified approach
        return 0.5 * (initial_displacement(x + c * t) + initial_displacement(x - c * t))

def wave_initial_condition_sine(x, amplitude=1.0, wavenumber=1.0, phase=0.0):
    """
    Generate sine wave initial condition: u(x,0) = A * sin(kx + φ)
    
    Args:
        x: spatial coordinate
        amplitude: wave amplitude A
        wavenumber: wave number k (2π/λ)
        phase: phase shift φ
    
    Returns:
        u_0: initial displacement
    """
    return amplitude * np.sin(wavenumber * x + phase)

def wave_initial_condition_gaussian(x, amplitude=1.0, center=0.5, width=0.1):
    """
    Generate Gaussian pulse initial condition: u(x,0) = A * exp(-(x-x₀)²/σ²)
    
    Args:
        x: spatial coordinate
        amplitude: pulse amplitude A
        center: center position x₀
        width: width parameter σ
    
    Returns:
        u_0: initial displacement
    """
    return amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2))

def wave_initial_velocity_zero(x):
    """
    Generate zero initial velocity: u_t(x,0) = 0
    
    Args:
        x: spatial coordinate
    
    Returns:
        u_t_0: initial velocity (all zeros)
    """
    return np.zeros_like(x)

def wave_initial_velocity_traveling(x, c, amplitude=1.0, wavenumber=1.0, phase=0.0):
    """
    Generate traveling wave initial velocity: u_t(x,0) = -c * u_x(x,0)
    This creates a wave that propagates to the right
    
    Args:
        x: spatial coordinate
        c: wave speed
        amplitude: wave amplitude A
        wavenumber: wave number k
        phase: phase shift φ
    
    Returns:
        u_t_0: initial velocity
    """
    # u_t(x,0) = -c * d/dx[A * sin(kx + φ)] = -c * A * k * cos(kx + φ)
    return -c * amplitude * wavenumber * np.cos(wavenumber * x + phase)

def wave_numerical_reference(x, t, c, initial_condition='sine', method='finite_difference'):
    """
    Generate numerical reference solution for wave equation.
    
    Args:
        x: spatial coordinate array
        t: time coordinate
        c: wave speed
        initial_condition: type of initial condition ('sine', 'gaussian', etc.)
        method: numerical method ('finite_difference', 'spectral')
    
    Returns:
        u: numerical reference solution
    """
    if method == 'finite_difference':
        # Simple finite difference solution for comparison
        nx = len(x)
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        
        # Set initial condition
        if initial_condition == 'sine':
            u = wave_initial_condition_sine(x, amplitude=1.0, wavenumber=2*np.pi)
        elif initial_condition == 'gaussian':
            u = wave_initial_condition_gaussian(x, amplitude=1.0, center=0.5, width=0.1)
        else:
            raise ValueError(f"Unknown initial condition: {initial_condition}")
        
        # Zero initial velocity
        u_t = wave_initial_velocity_zero(x)
        
        # Simple explicit time stepping
        dt = 0.001
        nt = int(t / dt)
        
        for n in range(nt):
            u_new = u.copy()
            for i in range(1, nx-1):
                # Finite difference approximation for wave equation
                u_xx = (u[i+1] - 2*u[i] + u[i-1]) / dx**2
                u_new[i] = u[i] + dt * u_t[i]
                u_t[i] = u_t[i] + dt * (c**2) * u_xx
            
            # Periodic boundary conditions
            u_new[0] = u_new[-2]
            u_new[-1] = u_new[1]
            u_t[0] = u_t[-2]
            u_t[-1] = u_t[1]
            u = u_new
        
        return u
    
    else:
        raise ValueError(f"Unknown numerical method: {method}") 