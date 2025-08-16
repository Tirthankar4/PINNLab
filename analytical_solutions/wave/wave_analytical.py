"""
wave_analytical.py
Analytical solutions for 1D wave equation for comparison with PINN results.
"""

import numpy as np
import torch

def wave_analytical_solution_dalembert(x, t, c, initial_displacement, initial_velocity=None, domain_length=1.0):
    """
    d'Alembert solution for the 1D wave equation with periodic boundary conditions: u_tt = c^2 * u_xx
    
    u(x,t) = (1/2)[f(x+ct) + f(x-ct)] + (1/(2c))∫_{x-ct}^{x+ct} g(ξ) dξ
    
    Args:
        x: spatial coordinate
        t: time coordinate  
        c: wave speed
        initial_displacement: function f(x) = u(x,0)
        initial_velocity: function g(x) = u_t(x,0) (optional, defaults to zero)
        domain_length: length of periodic domain (default 1.0 for [0,1])
    
    Returns:
        u: analytical solution
    """
    
    def periodic_function(func, x_eval, domain_length):
        """Apply periodic boundary conditions to function evaluation"""
        # Wrap x_eval to [0, domain_length] using modulo
        x_wrapped = x_eval % domain_length
        return func(x_wrapped)
    
    if initial_velocity is None:
        # If no initial velocity provided, assume g(x) = 0
        # Then u(x,t) = (1/2)[f(x+ct) + f(x-ct)]
        
        # Apply periodic boundary conditions to the shifted arguments
        f_forward = periodic_function(initial_displacement, x + c * t, domain_length)
        f_backward = periodic_function(initial_displacement, x - c * t, domain_length)
        
        return 0.5 * (f_forward + f_backward)
    else:
        # For non-zero initial velocity, compute both displacement and velocity terms
        f_forward = periodic_function(initial_displacement, x + c * t, domain_length)
        f_backward = periodic_function(initial_displacement, x - c * t, domain_length)
        
        # For the velocity integral term, we use a simplified approach:
        # ∫_{x-ct}^{x+ct} g(ξ) dξ ≈ 2ct * g(x) for smooth g(x)
        # This is exact for constant g(x) and a good approximation for slowly varying g(x)
        velocity_term = 2 * c * t * periodic_function(initial_velocity, x, domain_length) / (2 * c)
        
        return 0.5 * (f_forward + f_backward) + velocity_term

# Helper functions for numpy-based initial conditions (used only for analytical/numerical reference)
def _wave_initial_condition_sine_np(x, amplitude=1.0, wavenumber=1.0, phase=0.0):
    """Numpy version of sine initial condition for analytical solutions"""
    return amplitude * np.sin(wavenumber * x + phase)

def _wave_initial_condition_gaussian_np(x, amplitude=1.0, center=0.5, width=0.1):
    """Numpy version of Gaussian initial condition for analytical solutions"""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2))

def _wave_initial_velocity_zero_np(x):
    """Numpy version of zero initial velocity for analytical solutions"""
    return np.zeros_like(x)

def wave_analytical_solution_exact(x, t, c, initial_condition_type='sine', **kwargs):
    """
    Generate exact analytical solution for specific initial condition types.
    
    Args:
        x: spatial coordinate array
        t: time coordinate
        c: wave speed
        initial_condition_type: 'sine' or 'gaussian'
        **kwargs: additional parameters for initial conditions
    
    Returns:
        u: exact analytical solution
    """
    if initial_condition_type == 'sine':
        # For sine initial condition u(x,0) = sin(kx) with zero initial velocity
        # The exact solution is u(x,t) = sin(kx) * cos(kct) (standing wave)
        wavenumber = kwargs.get('wavenumber', 2*np.pi)
        amplitude = kwargs.get('amplitude', 1.0)
        
        # Handle scalar vs array inputs
        if np.isscalar(x) and np.isscalar(t):
            return amplitude * np.sin(wavenumber * x) * np.cos(wavenumber * c * t)
        elif np.isscalar(x):
            return np.array([amplitude * np.sin(wavenumber * x) * np.cos(wavenumber * c * ti) for ti in t])
        elif np.isscalar(t):
            return np.array([amplitude * np.sin(wavenumber * xi) * np.cos(wavenumber * c * t) for xi in x])
        else:
            # Both x and t are arrays - create meshgrid
            X, T = np.meshgrid(x, t, indexing='ij')
            result = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    result[i, j] = amplitude * np.sin(wavenumber * X[i, j]) * np.cos(wavenumber * c * T[i, j])
            return result
    
    elif initial_condition_type == 'gaussian':
        # For Gaussian initial condition, use d'Alembert solution with periodic wrapping
        amplitude = kwargs.get('amplitude', 1.0)
        center = kwargs.get('center', 0.5)
        width = kwargs.get('width', 0.1)
        domain_length = kwargs.get('domain_length', 1.0)
        
        def gaussian_initial(x_eval):
            return amplitude * np.exp(-((x_eval - center) ** 2) / (2 * width ** 2))
        
        return wave_analytical_solution_dalembert(x, t, c, gaussian_initial, 
                                                domain_length=domain_length)
    
    else:
        raise ValueError(f"Unknown initial condition type: {initial_condition_type}")

def wave_analytical_solution_traveling(x, t, c, initial_condition_type='sine', **kwargs):
    """
    Generate traveling wave solution for the wave equation.
    This provides better parameter dependency than standing waves.
    
    Args:
        x: spatial coordinate array
        t: time coordinate
        c: wave speed
        initial_condition_type: 'sine' or 'gaussian'
        **kwargs: additional parameters for initial conditions
    
    Returns:
        u: traveling wave solution
    """
    if initial_condition_type == 'sine':
        # For traveling wave: u(x,t) = sin(k(x - ct)) + sin(k(x + ct))
        # This shows clear wave speed dependency
        wavenumber = kwargs.get('wavenumber', 2*np.pi)
        amplitude = kwargs.get('amplitude', 1.0)
        
        # Handle scalar vs array inputs
        if np.isscalar(x) and np.isscalar(t):
            return amplitude * (np.sin(wavenumber * (x - c * t)) + np.sin(wavenumber * (x + c * t))) / 2
        elif np.isscalar(x):
            return np.array([amplitude * (np.sin(wavenumber * (x - c * ti)) + np.sin(wavenumber * (x + c * ti))) / 2 for ti in t])
        elif np.isscalar(t):
            return np.array([amplitude * (np.sin(wavenumber * (xi - c * t)) + np.sin(wavenumber * (xi + c * t))) / 2 for xi in x])
        else:
            # Both x and t are arrays - create meshgrid
            X, T = np.meshgrid(x, t, indexing='ij')
            result = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    result[i, j] = amplitude * (np.sin(wavenumber * (X[i, j] - c * T[i, j])) + np.sin(wavenumber * (X[i, j] + c * T[i, j]))) / 2
            return result
    
    else:
        raise ValueError(f"Traveling wave solution only implemented for sine initial condition")

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
            u = _wave_initial_condition_sine_np(x, amplitude=1.0, wavenumber=2*np.pi)
        elif initial_condition == 'gaussian':
            u = _wave_initial_condition_gaussian_np(x, amplitude=1.0, center=0.5, width=0.1)
        else:
            raise ValueError(f"Unknown initial condition: {initial_condition}")
        
        # Zero initial velocity
        u_t = _wave_initial_velocity_zero_np(x)
        
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