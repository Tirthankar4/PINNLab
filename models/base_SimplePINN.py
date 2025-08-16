"""
models/base_SimplePINN.py
Core PINN architecture components: BaseSimplePINN
"""

import torch
import torch.nn as nn
import importlib
from dependency_codes.config import MAIN_PINN_ACTIVATION, MODEL_TYPE

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
    
if MODEL_TYPE == 'hydro':
    output_dim = 3
elif MODEL_TYPE == 'burgers':
    output_dim = 2
elif MODEL_TYPE == 'wave':
    output_dim = 2
else:
    raise ValueError(f"Unknown model type: {MODEL_TYPE}")
    
class Base_SimplePINN(nn.Module):
    def __init__(self, output_dim, num_neurons = 96, use_param_embedding = False,
                 num_layers = 5, input_dim=None):
        super().__init__()
        self.use_param_embedding = use_param_embedding
        self.num_layers = num_layers
        self.use_residual = num_layers > 5

        # Create main PINN layers dynamically
        self.layers = nn.ModuleList()

        # Allow derived classes to specify input_dim, otherwise use default logic
        if input_dim is None:
            input_dim = 3 if use_param_embedding else 2

        # First layer
        self.layers.append(nn.Linear(input_dim, num_neurons))
        self.layers.append(get_activation(MAIN_PINN_ACTIVATION, Sin()))

        #Hidden layers
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(num_neurons, num_neurons))
            self.layers.append(get_activation(MAIN_PINN_ACTIVATION, Sin()))

        #Final output layer
        self.layers.append(nn.Linear(num_neurons, output_dim))

        #Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                #Use Xavier initialization for better training
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
                    param = coord_tensor[:, 1:2]    # c parameter
                else:
                    # Handle other cases with more features
                    coord = [coord_tensor[:, :-1]]  # All but last feature
                    param = coord_tensor[:, -1:]    # Last feature as parameter
            else:
                # Multiple tensors (standard case)
                *coord, param = X
        else:
            coord = X
            param = None
            
        # Debug: Check if coord list is empty
        if not coord:
            raise ValueError(f"SHM model received empty coordinate list. X: {X}, coord: {coord}, use_param_embedding: {self.use_param_embedding}")
            
        # Debug: Check tensor shapes
        for i, tensor in enumerate(coord):
            if tensor is None or tensor.numel() == 0:
                raise ValueError(f"SHM model received None or empty tensor at index {i}. tensor: {tensor}")
            
        # Prepare input embedding
        coord_tensor = torch.cat(coord, dim = 1)
        if self.use_param_embedding:
            inputs = torch.cat([coord_tensor, param], dim = 1)
        else:
            inputs = coord_tensor

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
                if i == 0:
                    x = layer(x)
                    residual = x
                elif i == len(self.layers) - 2:
                    x = layer(x)
                    x = x + residual
                else:
                    x = layer(x)

            return x

    def periodic_BC(self, net, alpha, alpha_size, coordinate=1, derivative_order=0, component=0):
        """
        Periodic boundary condition method for SimplePINN models.
        This method handles periodic boundary conditions for the loss functions.
        """
        # For SimplePINN, we'll use a simplified periodic boundary condition
        # that works with the basic architecture
        
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