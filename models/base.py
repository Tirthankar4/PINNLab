"""
models/base.py
Core PINN architecture components: BasePINN, ParameterEmbedding, InputEmbedding.
These are shared by all equation-specific PINN models.
"""

import torch
import torch.nn as nn
import importlib
from config import PARAM_EMBED_ACTIVATION, INPUT_EMBED_ACTIVATION, MAIN_PINN_ACTIVATION

class Sin(nn.Module):
    def forward(self, input):
        return torch.sin(input)

def get_activation(name, default):
    if name is None:
        return default
    name = name.lower()
    if name == 'tanh':
        return nn.Tanh()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'sin':
        return Sin()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'leakyrelu':
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unknown activation: {name}")

class ParameterEmbedding(nn.Module):
    def __init__(self, activation=None, num_layers=2, num_neurons=32):
        super().__init__()
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.use_residual = num_layers > 5
        
        # Create layers dynamically
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Linear(1, num_neurons))
        self.layers.append(get_activation(activation if activation is not None else PARAM_EMBED_ACTIVATION, nn.Tanh()))
        
        # Hidden layers
        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(num_neurons, num_neurons))
            self.layers.append(get_activation(activation if activation is not None else PARAM_EMBED_ACTIVATION, nn.Tanh()))
        
        # Final layer
        if num_layers > 1:
            self.layers.append(nn.Linear(num_neurons, num_neurons))
        
    def forward(self, alpha):
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(-1)
        elif alpha.dim() > 2:
            alpha = alpha.view(-1, 1)
        
        if not self.use_residual:
            # Standard forward pass
            x = alpha
            for layer in self.layers:
                x = layer(x)
            return x
        else:
            # Forward pass with residual connection
            x = alpha
            residual = None
            
            for i, layer in enumerate(self.layers):
                if i == 0:  # First linear layer
                    x = layer(x)
                    residual = x  # Store for residual connection
                elif i == len(self.layers) - 2:  # Second to last layer (before final linear)
                    x = layer(x)
                    x = x + residual  # Add residual connection
                else:
                    x = layer(x)
            
            return x

class InputEmbedding(nn.Module):
    def __init__(self, activation=None, num_layers=3, num_neurons=64):
        super().__init__()
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.use_residual = num_layers > 5
        
        # Input layers for different dimensions
        self.fc_in_1d = nn.Linear(2, num_neurons)
        self.fc_in_2d = nn.Linear(3, num_neurons)
        self.fc_in_3d = nn.Linear(4, num_neurons)
        
        # Create hidden layers dynamically
        self.layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(num_neurons, num_neurons))
            self.layers.append(get_activation(activation if activation is not None else INPUT_EMBED_ACTIVATION, Sin()))
        
        if num_layers > 1:
            self.layers.append(nn.Linear(num_neurons, num_neurons))
        
    def forward(self, coord):
        if coord.shape[1] == 2:
            x = self.fc_in_1d(coord)
        elif coord.shape[1] == 3:
            x = self.fc_in_2d(coord)
        elif coord.shape[1] == 4:
            x = self.fc_in_3d(coord)
        else:
            raise ValueError(f"Expected coord to have 2, 3, or 4 features, but got {coord.shape[1]} features with shape {coord.shape}")
        
        if not self.use_residual:
            # Standard forward pass
            for layer in self.layers:
                x = layer(x)
            return x
        else:
            # Forward pass with residual connection
            residual = x  # Store initial embedding for residual connection
            
            for i, layer in enumerate(self.layers):
                if i == len(self.layers) - 2:  # Second to last layer (before final linear)
                    x = layer(x)
                    x = x + residual  # Add residual connection
                else:
                    x = layer(x)
            
            return x

class BasePINN(nn.Module):
    def __init__(self, input_dim, output_dim, num_neurons=96, use_param_embedding=False,
                 param_embed_activation=None, input_embed_activation=None, main_pinn_activation=None,
                 param_embed_layers=2, param_embed_neurons=32, input_embed_layers=3, input_embed_neurons=64,
                 main_pinn_layers=5):
        super().__init__()
        self.use_param_embedding = use_param_embedding
        self.main_pinn_layers = main_pinn_layers
        self.use_residual = main_pinn_layers > 5
        
        if use_param_embedding:
            self.alpha_embedder = ParameterEmbedding(param_embed_activation, param_embed_layers, param_embed_neurons)
        
        self.input_embedder = InputEmbedding(input_embed_activation, input_embed_layers, input_embed_neurons)
        
        # Calculate input dimension for main PINN
        input_dim_main = (param_embed_neurons if use_param_embedding else 0) + input_embed_neurons
        
        # Create main PINN layers dynamically
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Linear(input_dim_main, num_neurons))
        self.layers.append(get_activation(main_pinn_activation if main_pinn_activation is not None 
                                        else MAIN_PINN_ACTIVATION, Sin()))
        
        # Hidden layers
        for i in range(main_pinn_layers - 1):
            self.layers.append(nn.Linear(num_neurons, num_neurons))
            self.layers.append(get_activation(main_pinn_activation if main_pinn_activation is not None 
                                            else MAIN_PINN_ACTIVATION, Sin()))
        
        # Final output layer
        self.layers.append(nn.Linear(num_neurons, output_dim))
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier initialization for better training
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, X):
        # X: list of tensors (x, t, [y, z, alpha, ...])
        # Determine if parameter embedding is used
        if self.use_param_embedding:
            *coord, alpha = X
        else:
            coord = X
            alpha = None
        # Prepare input embedding
        coord_tensor = torch.cat(coord, dim=1)
        input_emb = self.input_embedder(coord_tensor)
        if self.use_param_embedding:
            alpha_emb = self.alpha_embedder(alpha)
            inputs = torch.cat([input_emb, alpha_emb], dim=1)
        else:
            inputs = input_emb
        
        if not self.use_residual:
            # Standard forward pass
            x = inputs
            for layer in self.layers:
                x = layer(x)
            return x
        else:
            # Forward pass with residual connection
            x = inputs
            residual = None
            
            for i, layer in enumerate(self.layers):
                if i == 0:  # First linear layer
                    x = layer(x)
                    residual = x  # Store for residual connection
                elif i == len(self.layers) - 3:  # Second to last layer (before final linear)
                    x = layer(x)
                    x = x + residual  # Add residual connection
                else:
                    x = layer(x)
            
            return x 