import torch
import torch.nn as nn

class LegacyPINN(nn.Module):
    """Legacy PINN class compatible with saved models"""
    def __init__(self, num_neurons=96):
        super(LegacyPINN, self).__init__()

        # Alpha embedder (parameter embedding)
        self.alpha_embedder = nn.ModuleDict({
            'fc_in': nn.Linear(1, 32),
            'fc_out': nn.Linear(32, 32)
        })

        # Input embedder
        self.input_embedder = nn.ModuleDict({
            'fc_in_1d': nn.Linear(2, 64),
            'fc_in_2d': nn.Linear(3, 64),
            'fc_in_3d': nn.Linear(4, 64),
            'fc_hidden1': nn.Linear(64, 64),
            'fc_hidden2': nn.Linear(64, 64),
            'fc_out': nn.Linear(64, 64)
        })

        # Main network
        self.fc_in = nn.Linear(32 + 64, num_neurons)
        self.neurons1 = nn.Linear(num_neurons, num_neurons)
        self.neurons2 = nn.Linear(num_neurons, num_neurons)
        self.neurons3 = nn.Linear(num_neurons, num_neurons)
        self.neurons4 = nn.Linear(num_neurons, num_neurons)
        self.neurons5 = nn.Linear(num_neurons, num_neurons)

        self.fc_out_1d = nn.Linear(num_neurons, 3)
        self.fc_out_2d = nn.Linear(num_neurons, 4)
        self.fc_out_3d = nn.Linear(num_neurons, 5)

        self.sin_act = lambda x: torch.sin(x)

    def forward(self, X):
        # Extract and ensure correct dimensions
        if len(X) == 3:
            x, t, alpha = X[0], X[1], X[2]
            y, z = None, None
        elif len(X) == 4:
            x, y, t, alpha = X[0], X[1], X[2], X[3]
            z = None
        elif len(X) == 5:
            x, y, z, t, alpha = X[0], X[1], X[2], X[3], X[4]
        else:
            raise ValueError(f"Expected len(X) to be 3, 4 or 5 but got {len(X)}")
        
        # Ensure all inputs are [batch_size, 1]
        x = x if x.dim() == 2 else x.unsqueeze(-1)
        t = t if t.dim() == 2 else t.unsqueeze(-1)
        if y is not None:
            y = y if y.dim() == 2 else y.unsqueeze(-1)
        if z is not None:
            z = z if z.dim() == 2 else z.unsqueeze(-1)
        alpha = alpha if alpha.dim() == 2 else alpha.unsqueeze(-1)
        
        # Get batch size from x
        batch_size = x.size(0)
        
        # Process input coordinates
        if len(X) == 3:
            input_coord = torch.cat([x, t], dim=1)
        elif len(X) == 4:
            input_coord = torch.cat([x, y, t], dim=1)
        elif len(X) == 5:
            input_coord = torch.cat([x, y, z, t], dim=1)
        else:
            raise ValueError(f"Expected len(X) to be 3, 4 or 5 but got {len(X)}")

        # Input embedding
        if input_coord.shape[1] == 2:
            input_emb = self.input_embedder['fc_in_1d'](input_coord)
        elif input_coord.shape[1] == 3:
            input_emb = self.input_embedder['fc_in_2d'](input_coord)
        elif input_coord.shape[1] == 4:
            input_emb = self.input_embedder['fc_in_3d'](input_coord)
        else:
            raise ValueError(f"Unexpected input coordinate shape: {input_coord.shape}")
        
        input_emb = self.sin_act(input_emb)
        input_emb = self.sin_act(self.input_embedder['fc_hidden1'](input_emb))
        input_emb = self.sin_act(self.input_embedder['fc_hidden2'](input_emb))
        input_emb = self.input_embedder['fc_out'](input_emb)

        # Alpha embedding
        alpha_emb = self.sin_act(self.alpha_embedder['fc_in'](alpha))
        alpha_emb = self.alpha_embedder['fc_out'](alpha_emb)

        # Concatenate embeddings
        inputs = torch.cat([input_emb, alpha_emb], dim=1)

        # Process through network
        h0 = self.sin_act(self.fc_in(inputs))
        h1 = self.sin_act(self.neurons1(h0))
        h2 = self.sin_act(self.neurons2(h1))
        h3 = self.sin_act(self.neurons3(h2))
        h4 = self.sin_act(self.neurons4(h3)) + h0
        h5 = self.sin_act(self.neurons5(h4))

        # Select appropriate output layer
        if len(X) == 3:
            output = self.fc_out_1d(h5)
        elif len(X) == 4:
            output = self.fc_out_2d(h5)
        elif len(X) == 5:
            output = self.fc_out_3d(h5)
        else:
            raise ValueError(f"Expected len(X) to be 3, 4 or 5 but got {len(X)}")

        return output 