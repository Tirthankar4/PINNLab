"""
visualisation_burgers.py:
Visualization utilities for 1D Burgers' equation PINN solutions.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable

def plot_burgers_solution(net, time_array, initial_params, N, nu, isplot=True, true_solution_fn=None):
    """
    Plot the PINN solution for Burgers' equation at different times.
    net: trained PINN
    time_array: list/array of times to plot
    initial_params: (xmin, xmax, tmax, device)
    N: number of spatial points
    nu: viscosity
    isplot: whether to show the plot
    true_solution_fn: function u_true(x, t, nu) for comparison (optional)
    """
    xmin, xmax, tmax, device = initial_params
    res = N
    for t in time_array:
        X = np.linspace(xmin, xmax, res).reshape(res, 1)
        t_ = t * np.ones(res).reshape(res, 1)
        pt_x_collocation = Variable(torch.from_numpy(X).float(), requires_grad=True).to(device)
        pt_t_collocation = Variable(torch.from_numpy(t_).float(), requires_grad=True).to(device)
        
        # Always use parameter embedding for Burgers (like alpha in hydrodynamics)
        nu_input = torch.full((res, 1), nu, device=device, dtype=torch.float32)
        net_input = [pt_x_collocation, pt_t_collocation, nu_input]
            
        with torch.no_grad():
            u_pred = net(net_input).cpu().numpy()
        plt.plot(X, u_pred, label=f"PINN t={round(t,2)}")
        if true_solution_fn is not None:
            u_true = true_solution_fn(X, t, nu)
            plt.plot(X, u_true, '--', label=f"True t={round(t,2)}")
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.title("Burgers' Equation Solution")
    plt.legend()
    if isplot:
        plt.show()
    else:
        return X, u_pred 

def rel_misfit_burgers(net, time_array, initial_params, N, nu, true_solution_fn, show=True):
    """
    Plot the relative misfit between the PINN and true solution for Burgers' equation at different times.
    net: trained PINN
    time_array: list/array of times to plot
    initial_params: (xmin, xmax, tmax, device)
    N: number of spatial points
    nu: viscosity
    true_solution_fn: function u_true(x, t, nu) for comparison
    show: whether to show the plot
    """
    import matplotlib.pyplot as plt
    xmin, xmax, tmax, device = initial_params
    res = N
    fig, axes = plt.subplots(2, len(time_array), figsize=(4*len(time_array), 6))
    if len(time_array) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for j, t in enumerate(time_array):
        X = np.linspace(xmin, xmax, res).reshape(res, 1)
        t_ = t * np.ones(res).reshape(res, 1)
        pt_x_collocation = torch.from_numpy(X).float().to(device)
        pt_t_collocation = torch.from_numpy(t_).float().to(device)
        
        # Always use parameter embedding for Burgers (like alpha in hydrodynamics)
        nu_input = torch.full((res, 1), nu, device=device, dtype=torch.float32)
        net_input = [pt_x_collocation, pt_t_collocation, nu_input]
            
        with torch.no_grad():
            u_pred = net(net_input).cpu().numpy().flatten()
        u_true = true_solution_fn(X, t, nu).flatten()
        axes[0, j].plot(X, u_pred, label="PINN")
        axes[0, j].plot(X, u_true, '--', label="True")
        axes[0, j].set_title(f"t={t}")
        axes[0, j].set_xlabel("x")
        axes[0, j].set_ylabel("u(x, t)")
        axes[0, j].legend()
        # Absolute error instead of relative error
        abs_err = u_pred - u_true
        axes[1, j].plot(X, abs_err, color='black')
        axes[1, j].set_xlabel("x")
        axes[1, j].set_ylabel("Abs. Error")
        axes[1, j].set_title(f"Abs. Error t={t}")
        axes[1, j].grid(True)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        return axes 