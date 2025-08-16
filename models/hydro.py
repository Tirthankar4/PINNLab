"""
models/hydro.py
HydroPINN: PINN for 1D hydrodynamic system (rho, vx, phi) with parameter embedding.
"""

from .base_P2PINN import Base_P2PINN
from .base_SimplePINN import Base_SimplePINN

class HydroPINN(Base_P2PINN):
    def __init__(self, num_neurons=96, param_embed_activation=None, input_embed_activation=None, main_pinn_activation=None,
                 param_embed_layers=2, param_embed_neurons=32, input_embed_layers=3, input_embed_neurons=64,
                 main_pinn_layers=5):
        # input: (x, t, alpha), output: (rho, vx, phi)
        super().__init__(input_dim=3, output_dim=3, num_neurons=num_neurons, use_param_embedding=True,
                         param_embed_activation=param_embed_activation,
                         input_embed_activation=input_embed_activation,
                         main_pinn_activation=main_pinn_activation,
                         param_embed_layers=param_embed_layers,
                         param_embed_neurons=param_embed_neurons,
                         input_embed_layers=input_embed_layers,
                         input_embed_neurons=input_embed_neurons,
                         main_pinn_layers=main_pinn_layers)

class HydroSimplePINN(Base_SimplePINN):
    def __init__(self, num_neurons=96, use_param_embedding=True, num_layers=5):
        # input: (x, t, alpha), output: (rho, vx, phi)  
        super().__init__(output_dim=3, num_neurons=num_neurons, 
                        use_param_embedding=True, num_layers=num_layers) 