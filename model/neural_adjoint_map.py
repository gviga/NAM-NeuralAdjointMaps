import torch
import torch.nn as nn

class Neural_Adjoint_Map(nn.Module):
    '''
    This class defines a model composed of a linear module and a nonlinear module 
    to estimate a Neural Adjoint Map.
    '''
    def __init__(self, input_dim=128, output_dim=None, depth=4, width=128, act=nn.LeakyReLU(), bias=False, nonlinear_type="MLP"):
        super().__init__()

        # Define default output dimension if None
        if output_dim is None:
            output_dim = input_dim
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.width = width

        # Linear Module
        self.fmap_branch = nn.Linear(input_dim, output_dim, bias=bias)

        # Non-linear Module (could be different types: MLP, CNN, etc.)
        if nonlinear_type == "MLP":
            self.nonlinear_branch = self._build_mlp(input_dim, output_dim, width, depth, act, bias)
        elif nonlinear_type == "FC":
            self.nonlinear_branch = self._build_fully_connected(input_dim, output_dim, width, depth, act, bias)
        # TO DO: add additional non linear modules CNN, RNN, etc.
        
        # Apply small scaling to MLP output for initialization
        self.mlp_scale = 0.01

        # Initialize weights
        self._reset_parameters()

    def forward(self, x):
        '''
        Forward pass through both the linear and non-linear branches.
        '''
        verts = x[:, :self.input_dim]

        # Linear map
        fmap = self.fmap_branch(verts)

        # Nonlinear part
        t = self.mlp_scale * self.nonlinear_branch(verts)

        # Combine linear and nonlinear components
        x_out = fmap + t

        return x_out.squeeze(), fmap.squeeze(), t

    def _reset_parameters(self):
        '''
        Initialize the model parameters using Xavier uniform distribution.
        '''
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_mlp(self, input_dim, output_dim, width, depth, act, bias):
        '''
        Build an MLP (multi-layer perceptron) module.
        '''
        layers = []
        prev_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(prev_dim, width, bias=bias))
            layers.append(act)  # Add activation after each layer
            prev_dim = width
        layers.append(nn.Linear(prev_dim, output_dim, bias=bias))
        return nn.Sequential(*layers)

    def _build_fully_connected(self, input_dim, output_dim, width, depth, act, bias):
        '''
        Build a fully connected architecture if you need an alternative to MLP.
        '''
        # For example, a different structure of fully connected layers, if needed
        layers = []
        prev_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(prev_dim, width, bias=bias))
            layers.append(act)
            prev_dim = width
        layers.append(nn.Linear(prev_dim, output_dim, bias=bias))
        return nn.Sequential(*layers)


