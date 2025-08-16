"""
models/SHM.py
SHMPINN: PINN for 1D Simple Harmonic Motion equation with parameter embedding for spring constant.
"""

import torch
import torch.nn as nn
from .base_P2PINN import Base_P2PINN
from .base_SimplePINN import Base_SimplePINN

class SHMPINN(Base_P2PINN):
    def __init__(self, num_neurons=96, use_param_embedding=True,
                 param_embed_activation=None, input_embed_activation=None, main_pinn_activation=None,
                 param_embed_layers=2, param_embed_neurons=32, input_embed_layers=3, input_embed_neurons=64,
                 main_pinn_layers=5):
        # input: (t, c), output: x
        # Parameter embedding for spring constant (c)
        # For SHM: input_dim should be 2 (time + spring constant) when using parameter embedding
        # But the base class expects spatial dimensions, so we need to handle this differently
        if use_param_embedding:
            # With parameter embedding: [t, c] -> effective input_dim = 2
            effective_input_dim = 2
        else:
            # Without parameter embedding: [t] -> effective_input_dim = 1
            effective_input_dim = 1
            
        super().__init__(input_dim=effective_input_dim, output_dim=1, 
                        num_neurons=num_neurons, use_param_embedding=use_param_embedding,
                        param_embed_activation=param_embed_activation,
                        input_embed_activation=input_embed_activation,
                        main_pinn_activation=main_pinn_activation,
                        param_embed_layers=param_embed_layers,
                        param_embed_neurons=param_embed_neurons,
                        input_embed_layers=input_embed_layers,
                        input_embed_neurons=input_embed_neurons,
                        main_pinn_layers=main_pinn_layers)

class SHMSimplePINN(Base_SimplePINN):
    def __init__(self, num_neurons=96, use_param_embedding=True, num_layers=5):
        # input: (t, c), output: x
        # Parameter embedding for spring constant (c)  
        if use_param_embedding:
            # With parameter embedding: [t, c] -> input_dim = 2
            input_dim = 2
        else:
            # Without parameter embedding: [t] -> input_dim = 1
            input_dim = 1
            
        super().__init__(output_dim=1, num_neurons=num_neurons, 
                        use_param_embedding=use_param_embedding, num_layers=num_layers,
                        input_dim=input_dim)
