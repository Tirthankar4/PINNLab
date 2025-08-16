"""
visualisation_SHM.py
Visualization functions for Simple Harmonic Motion (SHM) equation solutions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from analytical_solutions.SHM import SHM_analytical_solution
from losses.SHM import SHM_initial_condition, SHM_initial_velocity

def plot_SHM_solution(net, time_array, initial_params, N, c, isplot=True, analytical_solution_fn=None):
    """
    Plot SHM equation solution using PINN model
    
    Parameters:
    -----------
    net : torch.nn.Module
        Trained PINN model
    time_array : numpy.ndarray
        Array of time points to plot
    initial_params : tuple
        (tmin, tmax, device) parameters
    N : int
        Number of time points
    c : float
        Spring constant parameter (ω = √c)
    isplot : bool
        Whether to display the plot
    analytical_solution_fn : function, optional
        Function to compute analytical solution for comparison
        
    Returns:
    --------
    dict : Dictionary containing plot data
    """
    tmin, tmax, device = initial_params
    
    # Create time grid
    T = np.linspace(tmin, tmax, N).reshape(N, 1)
    
    # Initialize arrays to store solutions
    x_pinn = np.zeros((len(time_array), N))
    x_analytical = np.zeros((len(time_array), N)) if analytical_solution_fn else None
    
    # Compute solutions across the full time grid T (sinusoidal curve expected)
    for i, _ in enumerate(time_array):
        # Prepare input for PINN across the full time grid
        pt_t_collocation = Variable(torch.from_numpy(T).float(), requires_grad=True).to(device)

        # Check if network uses parameter embedding
        if hasattr(net, 'use_param_embedding') and net.use_param_embedding:
            c_input = torch.full((N, 1), c, device=device, dtype=torch.float32)
            net_input = [pt_t_collocation, c_input]
        else:
            net_input = [pt_t_collocation]

        # Get PINN prediction for the entire curve x(t)
        with torch.no_grad():
            x_pred = net(net_input).cpu().numpy().flatten()

        # Check for very small values and scale if necessary
        if abs(x_pred.max()) < 1e-6 and abs(x_pred.min()) < 1e-6:
            x_pred = x_pred * 1e6

        x_pinn[i, :] = x_pred

        # Get analytical solution over T if provided
        if analytical_solution_fn:
            x_analytical[i, :] = analytical_solution_fn(T.flatten(), c, 1.0, 0.0)  # amplitude=1.0, initial_velocity=0.0
    
    # Create plots
    if isplot:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot PINN solution vs time for different amplitudes/initial conditions
        if len(time_array) == 1:
            # Single time point: just plot the one we have
            t_idx = 0
            axes[0].plot(T.flatten(), x_pinn[t_idx, :], linewidth=2, 
                        label=f't={time_array[t_idx]:.2f}')
        else:
            # Multiple time points: select representative times
            for i, t_idx in enumerate([0, len(time_array)//4, len(time_array)//2, 3*len(time_array)//4, -1]):
                if t_idx >= len(time_array):
                    continue
                axes[0].plot(T.flatten(), x_pinn[t_idx, :], linewidth=2, 
                            label=f't={time_array[t_idx]:.2f}')
        
        axes[0].set_xlabel('t')
        axes[0].set_ylabel('x(t)')
        axes[0].set_title(f'PINN SHM Solution (c={c}, ω={np.sqrt(c):.2f})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        if analytical_solution_fn:
            # Plot analytical solution comparison
            if len(time_array) == 1:
                # Single time point: just plot the one we have
                t_idx = 0
                axes[1].plot(T.flatten(), x_analytical[t_idx, :], '--', linewidth=2, 
                            label=f'Analytical t={time_array[t_idx]:.2f}')
                axes[1].plot(T.flatten(), x_pinn[t_idx, :], linewidth=2, 
                            label=f'PINN t={time_array[t_idx]:.2f}')
            else:
                # Multiple time points: select representative times
                for i, t_idx in enumerate([0, len(time_array)//4, len(time_array)//2, 3*len(time_array)//4, -1]):
                    if t_idx >= len(time_array):
                        continue
                    axes[1].plot(T.flatten(), x_analytical[t_idx, :], '--', linewidth=2, 
                                label=f'Analytical t={time_array[t_idx]:.2f}')
                    axes[1].plot(T.flatten(), x_pinn[t_idx, :], linewidth=2, 
                                label=f'PINN t={time_array[t_idx]:.2f}')
            
            axes[1].set_xlabel('t')
            axes[1].set_ylabel('x(t)')
            axes[1].set_title(f'PINN vs Analytical (c={c})')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            # Show phase space plot (x vs dx/dt) if possible
            axes[1].text(0.5, 0.5, 'No analytical solution\nprovided for comparison', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Comparison')
        
        plt.tight_layout()
        plt.show()
    
    return {
        'T': T,
        'time_array': time_array,
        'x_pinn': x_pinn,
        'x_analytical': x_analytical,
        'c': c
    }

def rel_misfit_SHM(net, time_array, initial_params, N, c, analytical_solution_fn=None, show=True):
    """
    Calculate and plot relative misfit between PINN and analytical solutions for SHM equation
    
    Parameters:
    -----------
    net : torch.nn.Module
        Trained PINN model
    time_array : numpy.ndarray
        Array of time points
    initial_params : tuple
        (tmin, tmax, device) parameters
    N : int
        Number of time points
    c : float
        Spring constant parameter
    analytical_solution_fn : function
        Function to compute analytical solution
    show : bool
        Whether to display the plot
        
    Returns:
    --------
    dict : Dictionary containing misfit data and plots
    """
    tmin, tmax, device = initial_params
    
    if analytical_solution_fn is None:
        # Use default analytical solution
        analytical_solution_fn = SHM_analytical_solution
    
    # Get solutions
    solutions = plot_SHM_solution(net, time_array, initial_params, N, c, isplot=False, 
                                 analytical_solution_fn=analytical_solution_fn)
    
    T = solutions['T']
    x_pinn = solutions['x_pinn']
    x_analytical = solutions['x_analytical']
    
    if x_analytical is None:
        raise ValueError("Analytical solution is required for misfit calculation")
    
    # Calculate relative misfit
    rel_misfit = np.zeros_like(x_pinn)
    for i in range(len(time_array)):
        for j in range(N):
            if abs(x_analytical[i, j]) > 1e-10:  # Avoid division by zero
                rel_misfit[i, j] = (x_pinn[i, j] - x_analytical[i, j]) / abs(x_analytical[i, j]) * 100
            else:
                rel_misfit[i, j] = 0.0
    
    # Calculate statistics
    mean_misfit = np.mean(np.abs(rel_misfit))
    max_misfit = np.max(np.abs(rel_misfit))
    rms_misfit = np.sqrt(np.mean(rel_misfit**2))
    
    if show:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot PINN vs Analytical at specific times
        if len(time_array) == 1:
            # Single time point: just use the one time we have
            plot_times = [time_array[0]]
            colors = ['blue']
        else:
            # Multiple time points: select representative times
            plot_times = [time_array[0], time_array[len(time_array)//2], time_array[-1]]
            colors = ['blue', 'red', 'green']
        
        for i, (t, color) in enumerate(zip(plot_times, colors)):
            t_idx = np.argmin(np.abs(time_array - t))
            axes[0, 0].plot(T, x_pinn[t_idx, :], color=color, linewidth=2, 
                           label=f'PINN t={t:.2f}')
            axes[0, 0].plot(T, x_analytical[t_idx, :], color=color, linestyle='--', 
                           linewidth=2, label=f'Analytical t={t:.2f}')
        
        axes[0, 0].set_xlabel('t')
        axes[0, 0].set_ylabel('x(t)')
        axes[0, 0].set_title(f'PINN vs Analytical SHM Solution (c={c})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot relative misfit
        # Ensure proper shapes for contourf: T should be 1D, rel_misfit should be 2D
        T_1d = T.flatten() if T.ndim > 1 else T
        # For single time point, we need to handle the 1D case properly
        if rel_misfit.shape[0] == 1:
            # Single time point: plot as line instead of contour
            axes[0, 1].plot(T_1d, rel_misfit[0, :], color='red', linewidth=2)
            axes[0, 1].set_xlabel('t')
            axes[0, 1].set_ylabel('Relative Misfit (%)')
            axes[0, 1].set_title('Relative Misfit at Single Time Point')
        else:
            # Multiple time points: use contourf
            im = axes[0, 1].contourf(T_1d, time_array, rel_misfit, levels=50, cmap='RdBu_r')
            axes[0, 1].set_xlabel('t')
            axes[0, 1].set_ylabel('Target Time')
            axes[0, 1].set_title('Relative Misfit (%)')
            plt.colorbar(im, ax=axes[0, 1])
        
        # Plot misfit statistics over time
        if rel_misfit.shape[0] == 1:
            # Single time point: show statistics as text
            mean_misfit_time = np.mean(np.abs(rel_misfit))
            max_misfit_time = np.max(np.abs(rel_misfit))
            axes[1, 0].text(0.5, 0.5, f'Single Time Point\nMean: {mean_misfit_time:.2f}%\nMax: {max_misfit_time:.2f}%', 
                           ha='center', va='center', transform=axes[1, 0].transAxes,
                           fontsize=12, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            axes[1, 0].set_title('Misfit Statistics (Single Time)')
            axes[1, 0].axis('off')
        else:
            # Multiple time points: plot statistics over time
            mean_misfit_time = np.mean(np.abs(rel_misfit), axis=1)
            max_misfit_time = np.max(np.abs(rel_misfit), axis=1)
            
            axes[1, 0].plot(time_array, mean_misfit_time, 'b-', linewidth=2, label='Mean')
            axes[1, 0].plot(time_array, max_misfit_time, 'r-', linewidth=2, label='Max')
            axes[1, 0].set_xlabel('Target Time')
            axes[1, 0].set_ylabel('Relative Misfit (%)')
            axes[1, 0].set_title('Misfit Statistics over Target Time')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Display statistics
        stats_text = f'Mean Misfit: {mean_misfit:.2f}%\n'
        stats_text += f'Max Misfit: {max_misfit:.2f}%\n'
        stats_text += f'RMS Misfit: {rms_misfit:.2f}%'
        
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].set_title('Misfit Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return {
        'T': T,
        'time_array': time_array,
        'x_pinn': x_pinn,
        'x_analytical': x_analytical,
        'rel_misfit': rel_misfit,
        'mean_misfit': mean_misfit,
        'max_misfit': max_misfit,
        'rms_misfit': rms_misfit,
        'c': c
    }

def plot_SHM_phase_space(net, time_array, initial_params, N, c, analytical_solution_fn=None):
    """
    Plot phase space diagram (x vs dx/dt) for SHM
    
    Parameters:
    -----------
    net : torch.nn.Module
        Trained PINN model
    time_array : numpy.ndarray
        Array of time points
    initial_params : tuple
        (tmin, tmax, device) parameters
    N : int
        Number of time points
    c : float
        Spring constant parameter
    analytical_solution_fn : function, optional
        Function to compute analytical solution
        
    Returns:
    --------
    dict : Dictionary containing phase space data
    """
    tmin, tmax, device = initial_params
    
    # Create a single time series for phase space plot
    T = np.linspace(tmin, tmax, N).reshape(N, 1)
    
    # Get PINN solution
    pt_t_collocation = Variable(torch.from_numpy(T).float(), requires_grad=True).to(device)
    
    if hasattr(net, 'use_param_embedding') and net.use_param_embedding:
        c_input = torch.full((N, 1), c, device=device, dtype=torch.float32)
        net_input = [pt_t_collocation, c_input]
    else:
        net_input = [pt_t_collocation]
    
    # Get position
    x_pred = net(net_input)
    
    # Get velocity (dx/dt) using automatic differentiation
    x_t = torch.autograd.grad(x_pred.sum(), pt_t_collocation, create_graph=True)[0]
    
    # Convert to numpy
    x_pinn = x_pred.detach().cpu().numpy().flatten()
    v_pinn = x_t.detach().cpu().numpy().flatten()
    
    # Get analytical solution if provided
    if analytical_solution_fn:
        x_analytical = analytical_solution_fn(T.flatten(), c, 1.0, 0.0)
        # Analytical velocity: dx/dt = -A*ω*sin(ωt) where x = A*cos(ωt)
        omega = np.sqrt(c)
        v_analytical = -1.0 * omega * np.sin(omega * T.flatten())
    else:
        x_analytical = None
        v_analytical = None
    
    # Create phase space plot
    plt.figure(figsize=(10, 8))
    
    # Plot PINN trajectory
    plt.plot(x_pinn, v_pinn, 'b-', linewidth=2, label='PINN')
    plt.scatter(x_pinn[0], v_pinn[0], color='blue', s=100, marker='o', label='Start (PINN)')
    plt.scatter(x_pinn[-1], v_pinn[-1], color='blue', s=100, marker='s', label='End (PINN)')
    
    if analytical_solution_fn:
        # Plot analytical trajectory
        plt.plot(x_analytical, v_analytical, 'r--', linewidth=2, label='Analytical')
        plt.scatter(x_analytical[0], v_analytical[0], color='red', s=100, marker='o', 
                   label='Start (Analytical)')
        plt.scatter(x_analytical[-1], v_analytical[-1], color='red', s=100, marker='s', 
                   label='End (Analytical)')
    
    plt.xlabel('Position x(t)')
    plt.ylabel('Velocity dx/dt')
    plt.title(f'SHM Phase Space (c={c}, ω={np.sqrt(c):.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()
    
    return {
        'T': T.flatten(),
        'x_pinn': x_pinn,
        'v_pinn': v_pinn,
        'x_analytical': x_analytical,
        'v_analytical': v_analytical,
        'c': c
    }

def plot_SHM_energy_conservation(net, time_array, initial_params, N, c):
    """
    Plot energy conservation for SHM equation
    
    Parameters:
    -----------
    net : torch.nn.Module
        Trained PINN model
    time_array : numpy.ndarray
        Array of time points
    initial_params : tuple
        (tmin, tmax, device) parameters
    N : int
        Number of time points
    c : float
        Spring constant parameter
        
    Returns:
    --------
    dict : Dictionary containing energy data
    """
    tmin, tmax, device = initial_params
    
    # Create time grid
    T = np.linspace(tmin, tmax, N).reshape(N, 1)
    
    # Get PINN solution with gradients
    pt_t_collocation = Variable(torch.from_numpy(T).float(), requires_grad=True).to(device)
    
    if hasattr(net, 'use_param_embedding') and net.use_param_embedding:
        c_input = torch.full((N, 1), c, device=device, dtype=torch.float32)
        net_input = [pt_t_collocation, c_input]
    else:
        net_input = [pt_t_collocation]
    
    # Get position and velocity
    x_pred = net(net_input)
    x_t = torch.autograd.grad(x_pred.sum(), pt_t_collocation, create_graph=True)[0]
    
    # Convert to numpy
    x = x_pred.detach().cpu().numpy().flatten()
    v = x_t.detach().cpu().numpy().flatten()
    
    # Calculate energy components
    # Kinetic energy: (1/2) * m * v^2 (assume m=1)
    # Potential energy: (1/2) * k * x^2 = (1/2) * c * x^2
    kinetic_energy = 0.5 * v**2
    potential_energy = 0.5 * c * x**2
    total_energy = kinetic_energy + potential_energy
    
    # Theoretical total energy (should be constant)
    # For x(t) = A*cos(ωt), E = (1/2)*c*A^2 = constant
    theoretical_energy = 0.5 * c * 1.0**2  # A=1.0
    
    # Plot energy components
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(T.flatten(), kinetic_energy, 'b-', linewidth=2, label='Kinetic')
    plt.plot(T.flatten(), potential_energy, 'r-', linewidth=2, label='Potential')
    plt.plot(T.flatten(), total_energy, 'k-', linewidth=2, label='Total')
    plt.axhline(y=theoretical_energy, color='gray', linestyle='--', label='Theoretical')
    plt.xlabel('t')
    plt.ylabel('Energy')
    plt.title(f'Energy Components (c={c})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot energy conservation error
    plt.subplot(2, 2, 2)
    energy_error = np.abs(total_energy - theoretical_energy) / theoretical_energy * 100
    plt.plot(T.flatten(), energy_error, 'r-', linewidth=2)
    plt.xlabel('t')
    plt.ylabel('Energy Conservation Error (%)')
    plt.title(f'Energy Conservation Error (c={c})')
    plt.grid(True, alpha=0.3)
    
    # Plot position and velocity
    plt.subplot(2, 2, 3)
    plt.plot(T.flatten(), x, 'b-', linewidth=2, label='Position x(t)')
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.title('Position vs Time')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(T.flatten(), v, 'r-', linewidth=2, label='Velocity dx/dt')
    plt.xlabel('t')
    plt.ylabel('dx/dt')
    plt.title('Velocity vs Time')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'T': T.flatten(),
        'x': x,
        'v': v,
        'kinetic_energy': kinetic_energy,
        'potential_energy': potential_energy,
        'total_energy': total_energy,
        'theoretical_energy': theoretical_energy,
        'energy_error': energy_error,
        'c': c
    }
