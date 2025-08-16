import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from analytical_solutions.wave import wave_analytical_solution_dalembert
from losses.wave import wave_initial_condition_sine, wave_initial_condition_gaussian

def plot_wave_solution(net, time_array, initial_params, N, c, isplot=True, analytical_solution_fn=None):
    """
    Plot wave equation solution using PINN model
    
    Parameters:
    -----------
    net : torch.nn.Module
        Trained PINN model
    time_array : numpy.ndarray
        Array of time points to plot
    initial_params : tuple
        (xmin, xmax, tmax, device) parameters
    N : int
        Number of spatial points
    c : float
        Wave speed parameter
    isplot : bool
        Whether to display the plot
    analytical_solution_fn : function, optional
        Function to compute analytical solution for comparison
        
    Returns:
    --------
    dict : Dictionary containing plot data
    """
    xmin, xmax, tmax, device = initial_params
    
    # Create spatial grid
    X = np.linspace(xmin, xmax, N).reshape(N, 1)
    
    # Initialize arrays to store solutions
    u_pinn = np.zeros((len(time_array), N))
    u_analytical = np.zeros((len(time_array), N)) if analytical_solution_fn else None
    
    # Compute solutions for each time
    for i, t in enumerate(time_array):
        t_ = t * np.ones(N).reshape(N, 1)
        
        # Prepare input for PINN
        pt_x_collocation = Variable(torch.from_numpy(X).float(), requires_grad=True).to(device)
        pt_t_collocation = Variable(torch.from_numpy(t_).float(), requires_grad=True).to(device)
        
        # Check if network uses parameter embedding
        if hasattr(net, 'use_param_embedding') and net.use_param_embedding:
            c_input = torch.full((N, 1), c, device=device, dtype=torch.float32)
            net_input = [pt_x_collocation, pt_t_collocation, c_input]
        else:
            net_input = [pt_x_collocation, pt_t_collocation]
        
        # Get PINN prediction
        with torch.no_grad():
            u_pred = net(net_input).cpu().numpy().flatten()
        
        # Check for very small values and scale if necessary
        if abs(u_pred.max()) < 1e-6 and abs(u_pred.min()) < 1e-6:
            u_pred = u_pred * 1e6
        
        u_pinn[i, :] = u_pred
        
        # Get analytical solution if provided
        if analytical_solution_fn:
            u_analytical[i, :] = analytical_solution_fn(X.flatten(), t, c)
    
    # Create plots
    if isplot:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot PINN solution
        im1 = axes[0].contourf(X.flatten(), time_array, u_pinn, levels=50, cmap='viridis')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('t')
        axes[0].set_title(f'PINN Solution (c={c})')
        plt.colorbar(im1, ax=axes[0])
        
        if analytical_solution_fn:
            # Plot analytical solution
            im2 = axes[1].contourf(X.flatten(), time_array, u_analytical, levels=50, cmap='viridis')
            axes[1].set_xlabel('x')
            axes[1].set_ylabel('t')
            axes[1].set_title(f'Analytical Solution (c={c})')
            plt.colorbar(im2, ax=axes[1])
        else:
            # Plot error or difference
            axes[1].text(0.5, 0.5, 'No analytical solution\nprovided for comparison', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Comparison')
        
        plt.tight_layout()
        plt.show()
    
    return {
        'X': X,
        'time_array': time_array,
        'u_pinn': u_pinn,
        'u_analytical': u_analytical,
        'c': c
    }

def rel_misfit_wave(net, time_array, initial_params, N, c, analytical_solution_fn=None, show=True):
    """
    Calculate and plot relative misfit between PINN and analytical solutions for wave equation
    
    Parameters:
    -----------
    net : torch.nn.Module
        Trained PINN model
    time_array : numpy.ndarray
        Array of time points
    initial_params : tuple
        (xmin, xmax, tmax, device) parameters
    N : int
        Number of spatial points
    c : float
        Wave speed parameter
    analytical_solution_fn : function
        Function to compute analytical solution
    show : bool
        Whether to display the plot
        
    Returns:
    --------
    dict : Dictionary containing misfit data and plots
    """
    xmin, xmax, tmax, device = initial_params
    
    if analytical_solution_fn is None:
        # Use default analytical solution with sine initial condition
        def default_analytical(x, t, c):
            return wave_analytical_solution_dalembert(x, t, c, wave_initial_condition_sine)
        analytical_solution_fn = default_analytical
    
    # Get solutions
    solutions = plot_wave_solution(net, time_array, initial_params, N, c, isplot=False, 
                                 analytical_solution_fn=analytical_solution_fn)
    
    X = solutions['X']
    u_pinn = solutions['u_pinn']
    u_analytical = solutions['u_analytical']
    
    if u_analytical is None:
        raise ValueError("Analytical solution is required for misfit calculation")
    
    # Calculate relative misfit
    rel_misfit = np.zeros_like(u_pinn)
    for i in range(len(time_array)):
        for j in range(N):
            if abs(u_analytical[i, j]) > 1e-10:  # Avoid division by zero
                rel_misfit[i, j] = (u_pinn[i, j] - u_analytical[i, j]) / abs(u_analytical[i, j]) * 100
            else:
                rel_misfit[i, j] = 0.0
    
    # Calculate statistics
    mean_misfit = np.mean(np.abs(rel_misfit))
    max_misfit = np.max(np.abs(rel_misfit))
    rms_misfit = np.sqrt(np.mean(rel_misfit**2))
    
    if show:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot PINN vs Analytical at specific times
        plot_times = [time_array[0], time_array[len(time_array)//2], time_array[-1]]
        colors = ['blue', 'red', 'green']
        
        for i, (t, color) in enumerate(zip(plot_times, colors)):
            t_idx = np.argmin(np.abs(time_array - t))
            axes[0, 0].plot(X, u_pinn[t_idx, :], color=color, linewidth=2, 
                           label=f'PINN t={t:.2f}')
            axes[0, 0].plot(X, u_analytical[t_idx, :], color=color, linestyle='--', 
                           linewidth=2, label=f'Analytical t={t:.2f}')
        
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('u(x, t)')
        axes[0, 0].set_title(f'PINN vs Analytical Solution (c={c})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot relative misfit
        if len(time_array) == 1:
            # Single time point: plot as line instead of contour
            axes[0, 1].plot(X.flatten(), rel_misfit[0, :], color='red', linewidth=2)
            axes[0, 1].set_xlabel('x')
            axes[0, 1].set_ylabel('Relative Misfit (%)')
            axes[0, 1].set_title('Relative Misfit at Single Time Point')
        else:
            # Multiple time points: use contourf
            im = axes[0, 1].contourf(X.flatten(), time_array, rel_misfit, levels=50, cmap='RdBu_r')
            axes[0, 1].set_xlabel('x')
            axes[0, 1].set_ylabel('t')
            axes[0, 1].set_title('Relative Misfit (%)')
            plt.colorbar(im, ax=axes[0, 1])
        
        # Plot misfit statistics over time
        if len(time_array) == 1:
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
            axes[1, 0].set_xlabel('t')
            axes[1, 0].set_ylabel('Relative Misfit (%)')
            axes[1, 0].set_title('Misfit Statistics over Time')
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
        'X': X,
        'time_array': time_array,
        'u_pinn': u_pinn,
        'u_analytical': u_analytical,
        'rel_misfit': rel_misfit,
        'mean_misfit': mean_misfit,
        'max_misfit': max_misfit,
        'rms_misfit': rms_misfit,
        'c': c
    }

def plot_wave_animation(net, time_array, initial_params, N, c, analytical_solution_fn=None, save_path=None):
    """
    Create animation of wave equation solution
    
    Parameters:
    -----------
    net : torch.nn.Module
        Trained PINN model
    time_array : numpy.ndarray
        Array of time points
    initial_params : tuple
        (xmin, xmax, tmax, device) parameters
    N : int
        Number of spatial points
    c : float
        Wave speed parameter
    analytical_solution_fn : function, optional
        Function to compute analytical solution
    save_path : str, optional
        Path to save animation
        
    Returns:
    --------
    matplotlib.animation.Animation : Animation object
    """
    from matplotlib.animation import FuncAnimation
    
    xmin, xmax, tmax, device = initial_params
    
    # Get solutions
    solutions = plot_wave_solution(net, time_array, initial_params, N, c, isplot=False, 
                                 analytical_solution_fn=analytical_solution_fn)
    
    X = solutions['X']
    u_pinn = solutions['u_pinn']
    u_analytical = solutions['u_analytical']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Initialize lines
    line_pinn, = ax.plot([], [], 'b-', linewidth=2, label='PINN')
    if u_analytical is not None:
        line_analytical, = ax.plot([], [], 'r--', linewidth=2, label='Analytical')
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(np.min(u_pinn) - 0.1, np.max(u_pinn) + 0.1)
    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t)')
    ax.set_title(f'Wave Equation Solution (c={c})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    def animate(frame):
        line_pinn.set_data(X.flatten(), u_pinn[frame, :])
        if u_analytical is not None:
            line_analytical.set_data(X.flatten(), u_analytical[frame, :])
        ax.set_title(f'Wave Equation Solution (c={c}) - t={time_array[frame]:.3f}')
        return line_pinn, line_analytical if u_analytical is not None else line_pinn
    
    anim = FuncAnimation(fig, animate, frames=len(time_array), interval=100, blit=True)
    
    if save_path:
        anim.save(save_path, writer='pillow')
    
    return anim

def plot_wave_energy_conservation(net, time_array, initial_params, N, c):
    """
    Plot energy conservation for wave equation
    
    Parameters:
    -----------
    net : torch.nn.Module
        Trained PINN model
    time_array : numpy.ndarray
        Array of time points
    initial_params : tuple
        (xmin, xmax, tmax, device) parameters
    N : int
        Number of spatial points
    c : float
        Wave speed parameter
        
    Returns:
    --------
    dict : Dictionary containing energy data
    """
    xmin, xmax, tmax, device = initial_params
    
    # Get solutions
    solutions = plot_wave_solution(net, time_array, initial_params, N, c, isplot=False)
    
    X = solutions['X']
    u_pinn = solutions['u_pinn']
    
    # Calculate energy (kinetic + potential)
    dx = (xmax - xmin) / (N - 1)
    energy = np.zeros(len(time_array))
    
    for i, t in enumerate(time_array):
        # Kinetic energy: (1/2) * (du/dt)^2
        # Potential energy: (1/2) * c^2 * (du/dx)^2
        
        # Approximate derivatives using finite differences
        if i > 0 and i < len(time_array) - 1:
            du_dt = (u_pinn[i+1, :] - u_pinn[i-1, :]) / (2 * (time_array[1] - time_array[0]))
        elif i == 0:
            du_dt = (u_pinn[i+1, :] - u_pinn[i, :]) / (time_array[1] - time_array[0])
        else:
            du_dt = (u_pinn[i, :] - u_pinn[i-1, :]) / (time_array[1] - time_array[0])
        
        du_dx = np.gradient(u_pinn[i, :], dx)
        
        kinetic_energy = 0.5 * np.sum(du_dt**2) * dx
        potential_energy = 0.5 * c**2 * np.sum(du_dx**2) * dx
        
        energy[i] = kinetic_energy + potential_energy
    
    # Plot energy conservation
    plt.figure(figsize=(10, 6))
    plt.plot(time_array, energy, 'b-', linewidth=2)
    plt.xlabel('t')
    plt.ylabel('Total Energy')
    plt.title(f'Energy Conservation (c={c})')
    plt.grid(True, alpha=0.3)
    
    # Calculate energy conservation error
    initial_energy = energy[0]
    energy_error = np.abs(energy - initial_energy) / initial_energy * 100
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_array, energy_error, 'r-', linewidth=2)
    plt.xlabel('t')
    plt.ylabel('Energy Conservation Error (%)')
    plt.title(f'Energy Conservation Error (c={c})')
    plt.grid(True, alpha=0.3)
    
    plt.show()
    
    return {
        'time_array': time_array,
        'energy': energy,
        'energy_error': energy_error,
        'initial_energy': initial_energy,
        'c': c
    } 