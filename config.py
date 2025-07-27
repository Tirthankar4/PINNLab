MODEL_TYPE = 'hydro'  # Options: 'hydro', 'burgers'

tmin = 0.
xmin = 0.
ymin= 0.
zmin= 0.

cs = 1.0
rho_o = 1
const = 1
G = 1


iteration_adam_1D = 500
iteration_lbgfs_1D = 51

iteration_adam_2D = 1000
iteration_lbgfs_2D = 201

iteration_adam_3D = 10
iteration_lbgfs_3D = 2

# Activation function options (set to None to use defaults in code)
PARAM_EMBED_ACTIVATION = None  # e.g., 'tanh', 'relu', 'sin', etc.
INPUT_EMBED_ACTIVATION = None  # e.g., 'sin', 'relu', etc.
MAIN_PINN_ACTIVATION = None    # e.g., 'sin', 'relu', etc.
