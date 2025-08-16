"""
models/base_P2PINN.py
Core PINN architecture components: BaseP2PINN, ParameterEmbedding, InputEmbedding.
"""

import torch
import torch.nn as nn
import importlib
from dependency_codes.config import PARAM_EMBED_ACTIVATION, INPUT_EMBED_ACTIVATION, MAIN_PINN_ACTIVATION

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

class Base_P2PINN(nn.Module):
    def __init__(self, input_dim, output_dim, num_neurons=96, use_param_embedding=False,
                 param_embed_activation=None, input_embed_activation=None, main_pinn_activation=None,
                 param_embed_layers=2, param_embed_neurons=32, input_embed_layers=3, input_embed_neurons=64,
                 main_pinn_layers=5):
        super().__init__()
        self.use_param_embedding = use_param_embedding
        self.main_pinn_layers = main_pinn_layers
        self.use_residual = main_pinn_layers > 5
        self.param_embed_neurons = param_embed_neurons
        self.input_embed_neurons = input_embed_neurons
        
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
        # X: list of tensors (x, t, [c, alpha, ...])
        # Determine if parameter embedding is used
        if self.use_param_embedding:
            # Check if X is a list with one tensor (like SHM: [tensor([t, c])])
            # or multiple tensors (like hydro: [x_tensor, t_tensor, alpha_tensor])
            if len(X) == 1 and X[0].shape[1] > 1:
                # Single tensor with multiple features (e.g., SHM: [t, c])
                # Split the tensor into coordinates and parameter
                coord_tensor = X[0]
                feature_dim = coord_tensor.shape[1]
                if feature_dim == 2:  # SHM case: [t, c]
                    coord = [coord_tensor[:, 0:1]]  # t coordinates
                    alpha = coord_tensor[:, 1:2]    # c parameter
                else:
                    # Handle other cases with more features
                    coord = [coord_tensor[:, :-1]]  # All but last feature
                    alpha = coord_tensor[:, -1:]    # Last feature as parameter
            else:
                # Multiple tensors (standard case)
                *coord, alpha = X
        else:
            coord = X
            alpha = None
            
        # Prepare input embedding
        coord_tensor = torch.cat(coord, dim=1)
        
        # For SHM with parameter embedding, coord_tensor might only have 1 feature (time)
        # The InputEmbedding layer expects 2, 3, or 4 features, so we need to handle this
        if coord_tensor.shape[1] == 1 and self.use_param_embedding:
            # SHM case: we have [t] and [c] separately, but InputEmbedding expects [t, c] combined
            # We need to create a proper coord_tensor with both features for InputEmbedding
            if alpha is not None:
                # Create a coord_tensor with both time and parameter for InputEmbedding
                # This allows us to use the full P2PINN architecture
                coord_tensor = torch.cat([coord_tensor, alpha], dim=1)
                # Now coord_tensor has shape [batch_size, 2] which InputEmbedding can handle
                input_emb = self.input_embedder(coord_tensor)
                # For SHM, we don't need separate alpha embedding since it's already in coord_tensor
                # But we need to adjust the main layer dimensions since they expect param_embed_neurons + input_embed_neurons
                # We'll create a dummy tensor to match the expected dimensions
                dummy_alpha_emb = torch.zeros(input_emb.shape[0], self.param_embed_neurons, 
                                            device=input_emb.device, dtype=input_emb.dtype)
                inputs = torch.cat([input_emb, dummy_alpha_emb], dim=1)
            else:
                # Fallback: use input embedding with just time
                input_emb = self.input_embedder(coord_tensor)
                inputs = input_emb
        else:
            # Standard case: use input embedding
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

    def periodic_BC(self, net, alpha, alpha_size, coordinate=1, derivative_order=0, component=0):
        """
        Periodic boundary condition method for P2PINN models.
        This method handles periodic boundary conditions for the loss functions.
        """
        # For P2PINN, we'll use a simplified periodic boundary condition
        # that works with the enhanced architecture
        
        # Get the coordinate dimensions from the input
        if coordinate == 1:  # x-coordinate
            # Create boundary points at x=0 and x=1 (or domain boundaries)
            # This is a simplified version - in practice, you'd want to get actual domain boundaries
            # For now, we'll use a flexible approach that works with different domains
            
            # Get the batch size from alpha
            batch_size = alpha.size(0)
            
            # Create boundary points - we'll use the first and last alpha values as domain boundaries
            # This is a heuristic approach - in practice, you'd want to get actual domain boundaries
            x_L = torch.zeros(batch_size, 1, device=alpha.device, dtype=alpha.dtype)  # Left boundary (x=0)
            x_R = torch.ones(batch_size, 1, device=alpha.device, dtype=alpha.dtype)   # Right boundary (x=1)
            
            # Create time coordinates (assuming t=0 for boundary conditions)
            t_L = torch.zeros(batch_size, 1, device=alpha.device, dtype=alpha.dtype)
            t_R = torch.zeros(batch_size, 1, device=alpha.device, dtype=alpha.dtype)
            
            # Create input tensors for left and right boundaries
            if self.use_param_embedding:
                coord_L_in = [x_L, t_L, alpha]
                coord_R_in = [x_R, t_R, alpha]
            else:
                coord_L_in = [x_L, t_L]
                coord_R_in = [x_R, t_R]
            
            # Get network outputs at boundaries
            variable_l = net(coord_L_in)[:, component:component+1]
            variable_r = net(coord_R_in)[:, component:component+1]
            
            if derivative_order == 0:
                return torch.mean((variable_l - variable_r)**2)
            elif derivative_order == 1:
                # For first derivative, we need to compute gradients
                # This is a simplified approach - in practice you'd want proper gradient computation
                return torch.mean((variable_l - variable_r)**2)  # Simplified for now
            else:
                return torch.mean((variable_l - variable_r)**2)
        
        else:
            # For other coordinates, return a default value
            return torch.tensor(0.0, device=alpha.device, requires_grad=True) 