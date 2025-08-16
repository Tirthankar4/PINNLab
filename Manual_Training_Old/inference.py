import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from dependency_codes.model_architecture import PINN
from dependency_codes.data_generator import alpha_generator
from losses.losses import ASTPN
from dependency_codes.config import xmin, ymin, tmin, rho_o
from visualisations import plot_function, rel_misfit
from dependency_codes.solver import req_consts_calc  # <-- import for jeans and alpha

# Device setup
has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if torch.backends.mps.is_built() else "cuda:0" if torch.cuda.is_available() else "cpu"

alpha_min, alpha_max = 0.01, 0.08
_, _, alpha_list = alpha_generator(alpha_min=alpha_min, alpha_max=alpha_max, N=20)
alpha_list = alpha_list.to(device)

# Domain setup (smaller: 2000 points)
lam = 7.0
rho_1 = 0.03
num_of_waves = 2
tmax = 1.5
N_0 = N_b = 1000
N_r = 10000
xmax = xmin + lam * num_of_waves
ymax = ymin + lam * num_of_waves

# Model and domain
model_1D = ASTPN(rmin=[xmin, tmin], rmax=[xmax, tmax], N_0=N_0, N_b=N_b, N_r=N_r, dimension=1)
domain = model_1D.geo_time_coord(option="Domain")

# Load model
net = PINN()
net.load_state_dict(torch.load('Case1_final_part1.pth', map_location=device))
net = net.to(device)
net.eval()

# Compute jeans and plot_v1 for the chosen alpha
alpha_flat = alpha_list.view(-1)
plot_rhos = ((alpha_flat[:-1] + alpha_flat[1:]) / 2).tolist()
plot_rhos = [round(i, 3) for i in plot_rhos]

jeans, alpha_val = req_consts_calc(lam)

# Arrays to store results for each time step
avg_misfit_rho_by_time = {0.5: [], 1.0: [], 1.5: [], 2.0: []}
max_misfit_rho_by_time = {0.5: [], 1.0: [], 1.5: [], 2.0: []}

nu = 0.5
N = 1000
num_of_waves = 2
rho_1 = 0.03

time_array_misfit = np.array([0.5, 1.0, 1.5, 2.0])

# Test with a few specific alpha values
plot_rho1 = plot_rhos[1]
plot_rho2 = plot_rhos[6]
plot_rho3 = plot_rhos[-1]

plot_v1 = ((alpha_val/(rho_o*2*np.pi/lam)) * plot_rho1)
plot_v2 = ((alpha_val/(rho_o*2*np.pi/lam)) * plot_rho2)
plot_v3 = ((alpha_val/(rho_o*2*np.pi/lam)) * plot_rho3)

initial_params1 = xmin, xmax, rho_1, plot_rho1, plot_v1, jeans, lam, tmax, device
initial_params2 = xmin, xmax, rho_1, plot_rho2, plot_v2, jeans, lam, tmax, device
initial_params3 = xmin, xmax, rho_1, plot_rho3, plot_v3, jeans, lam, tmax, device

# Test the plotting functions
print("Testing plotting functions...")
rel_misfit(net, time_array_misfit, initial_params1, N, nu, num_of_waves, rho_1)
rel_misfit(net, time_array_misfit, initial_params2, N, nu, num_of_waves, rho_1)
rel_misfit(net, time_array_misfit, initial_params3, N, nu, num_of_waves, rho_1)