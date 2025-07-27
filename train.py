import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
#torch.cuda.empty_cache()
import time

from data_generator import alpha_generator
from solver import input_taker, req_consts_calc, closure_batched, train_batched
from config import xmin, tmin, iteration_adam_1D, iteration_lbgfs_1D
from config import rho_o
from losses.losses import ASTPN
from models.burgers import BurgersPINN
from visualisation import plot_function, rel_misfit
from config import MODEL_TYPE
from visualisation_burgers import plot_burgers_solution
from visualisation_burgers import rel_misfit_burgers

# Define or import a true solution function for Burgers' equation:
def true_solution_fn(x, t, nu):
    """
    Analytical solution for Burgers' equation with u(x,0) = -sin(πx)
    and periodic boundary conditions.
    """
    # This is the exact solution for this specific case
    return -np.sin(np.pi * x) * np.exp(-nu * np.pi**2 * t)

has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if torch.backends.mps.is_built() else "cuda:0" if torch.cuda.is_available() else "cpu"

alpha_min, alpha_max, alpha_list = alpha_generator(alpha_min = 0.01, alpha_max = 0.1, N = 5)
alpha_list = alpha_list.to(device)

# Domain configuration - can be made configurable
N_0_default, N_b_default, N_r_default = 100, 100, 1000
lam, rho_1, num_of_waves, tmax, N_0, N_b, N_r = input_taker(7.0, 0.03, 2, 1.5, N_0_default, N_b_default, N_r_default)

jeans, alpha = req_consts_calc(lam)
#v_1  = (rho_1/rho_o) * (alpha/(2*np.pi/lam))

# Set xmax based on model type
if MODEL_TYPE == 'hydro':
    xmax = xmin + lam * num_of_waves
elif MODEL_TYPE == 'burgers':
    xmax = 1.0  # Burgers uses [0,1] domain for sin(pi*x) initial condition
else:
    raise ValueError(f'Unknown MODEL_TYPE: {MODEL_TYPE}')

if MODEL_TYPE == 'hydro':
    from models.hydro import HydroPINN
    net = HydroPINN()
    print('Training HydroPINN (hydrodynamics)')
elif MODEL_TYPE == 'burgers':
    from models.burgers import BurgersPINN
    net = BurgersPINN()
    print('Training BurgersPINN (Burgers\' equation) with parameter embedding')
else:
    raise ValueError(f'Unknown MODEL_TYPE: {MODEL_TYPE}')

net = net.to(device)

mse_cost_function = torch.nn.MSELoss() # Mean squared error
optimizer = torch.optim.Adam(net.parameters(),lr=0.001,)
optimizerL = torch.optim.LBFGS(net.parameters(),line_search_fn='strong_wolfe')

model_1D = ASTPN(rmin = [xmin, tmin], rmax = [xmax, tmax], 
                 N_0 = N_0, N_b = N_b, N_r = N_r, dimension = 1)

collocation_domain_1D = model_1D.geo_time_coord(option = "Domain") 
collocation_IC_1D = model_1D.geo_time_coord(option = "IC")

#print("alpha_list: ", alpha_list)

v_1 = ((alpha/(rho_o*2*np.pi/lam)) * alpha_list)
#print("v1: ", v_1)

start_time = time.time()
train_batched(
    net=net,
    model=model_1D,
    alpha=alpha_list,
    collocation_domain=collocation_domain_1D,
    collocation_IC=collocation_IC_1D,
    optimizer=optimizer,
    optimizerL=optimizerL,
    closure=closure_batched,
    mse_cost_function=mse_cost_function,
    iteration_adam=iteration_adam_1D,
    iterationL=iteration_lbgfs_1D,
    lam=lam,
    jeans=jeans,
    v_1=v_1,
    batch_size = 1000,
    num_batches = 5
)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

plot_rho = 1.1
plot_rho_1 = 2.1

plot_v1 = ((alpha/(rho_o*2*np.pi/lam)) * plot_rho)
plot_v1_1 = ((alpha/(rho_o*2*np.pi/lam)) * plot_rho_1)

time_array_plot = np.linspace(0.5,int(tmax),int(tmax)+3)
time_array_misfit = np.array([0.5, 1.0, 1.5, 2.0, 2.5])

initial_params_1 = xmin, xmax, rho_1, plot_rho, plot_v1, jeans, lam, tmax, device
initial_params_2 = xmin, xmax, rho_1, plot_rho_1, plot_v1_1, jeans, lam, tmax, device

nu = 0.5
N = 1000

'''plot_function(net,time_array_plot,initial_params_1, N, velocity=True,isplot =True)
plot_function(net,time_array_plot,initial_params_2, N, velocity=True,isplot =True)'''

if MODEL_TYPE == 'hydro':
    rel_misfit(net, time_array_misfit, initial_params_1, N, nu, num_of_waves, rho_1)
    rel_misfit(net, time_array_misfit, initial_params_2, N, nu, num_of_waves, rho_1)
elif MODEL_TYPE == 'burgers':
    initial_params = (xmin, xmax, tmax, device)
    # Test the true solution function
    '''print("Testing true solution function:")
    test_x = np.array([[0.5], [1.0], [2.0]])
    test_t = 0.5
    test_nu = 0.02
    test_result = true_solution_fn(test_x, test_t, test_nu)
    print(f"x={test_x.flatten()}, t={test_t}, nu={test_nu}")
    print(f"True solution: {test_result.flatten()}")
    print(f"Expected decay factor: exp(-{test_nu}*pi^2*{test_t}) = {np.exp(-test_nu * np.pi**2 * test_t):.4f}")'''
    
    # Plot for two different nu values from the training range
    rel_misfit_burgers(net, [0.0, 0.5, 1.0], initial_params, N=1000, nu=0.02, true_solution_fn=true_solution_fn)
    plt.show()  # Ensure first plot is displayed
    rel_misfit_burgers(net, [0.0, 0.5, 1.0], initial_params, N=1000, nu=0.08, true_solution_fn=true_solution_fn)
    plt.show()  # Ensure second plot is displayed

'''# Save the trained model
MODEL_PATH = 'Running_Final_pinn_model.pth'
torch.save(net.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")'''