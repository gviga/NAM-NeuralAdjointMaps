import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# defien the neural map module
class HybridModel(nn.Module):
    """
    This class defines a model composed of a linear module and a nonlinear module
    to estimate a Neural Adjoint Map.

    Parameters:
    ----------
    input_dim : int
        The dimension of the input data.
    output_dim : int
        The dimension of the output data. If None, it defaults to input_dim.
    depth : int
        The number of layers in the MLP.
    width : int
        The width of each layer in the MLP.
    act : torch.nn.Module
        The activation function to be used in the MLP.

    """

    def __init__(
        self,
        input_dim,
        output_dim,
        depth=4,
        width=128,
        act=nn.LeakyReLU(),
    ):
        super().__init__()

        # Define default output dimension if None
        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.width = width

        # Linear Module
        self.fmap_branch = nn.Linear(input_dim, output_dim, bias=False)

        self.nonlinear_branch = self._build_mlp(
            input_dim, output_dim, width, depth, act, False
        )

        # Apply small scaling to MLP output for initialization
        self.mlp_scale = 0.01

        # Initialize weights
        self._reset_parameters()

    def forward(self, x):
        """
        Forward pass through both the linear and non-linear branches.
        """
        verts = x[:, : self.input_dim]

        # Linear map
        fmap = self.fmap_branch(verts)

        # Nonlinear part
        t = self.mlp_scale * self.nonlinear_branch(verts)

        # Combine linear and nonlinear components
        x_out = fmap + t

        return x_out.squeeze()

    def _reset_parameters(self):
        """
        Initialize the model parameters using Xavier uniform distribution.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_mlp(self, input_dim, output_dim, width, depth, act, bias):
        """
        Build an MLP (multi-layer perceptron) module.
        """
        layers = []
        prev_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(prev_dim, width, bias=bias))
            layers.append(act)  # Add activation after each layer
            prev_dim = width
        layers.append(nn.Linear(prev_dim, output_dim, bias=bias))
        return nn.Sequential(*layers)


class NeuralAdjointMap(nn.Module):
    """
    This class defines a neural adjoint map between a pair of embedding.

    Parameters:
    ----------
    embedding1 : torch.Tensor
        The first embedding tensor.
    embedding2 : torch.Tensor
        The second embedding tensor.
    """

    def __init__(self, embedding1, embedding2, model=None, optimizer=None):
        super().__init__()

        self.device = embedding1.device

        self.embedding1 = embedding1
        self.embedding2 = embedding2
        self.optimizer = optimizer
        self.model = model

        if self.model is None:
            self.model = HybridModel(
                input_dim=embedding2.shape[-1],
                output_dim=embedding1.shape[-1],
                width=128,
                depth=2,
                act=nn.LeakyReLU(),
            ).to(self.device)

        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.1
        )

        self.optimized = False

    def optimize_from_p2p(self, p2p, iter_max=200, patience=10, min_delta=1e-4):
        if self.optimized:
            print(
                "Model already optimized. If you want to re-optimize, please set self.optimized to False."
            )

        else:
            best_loss = float("inf")
            wait = 0

            for _ in range(iter_max):
                self.optimizer.zero_grad()

                pred = self.model(self.embedding2)

                loss = F.mse_loss(pred[p2p], self.embedding1)
                loss.backward()
                self.optimizer.step()

                if loss.item() < best_loss - min_delta:
                    best_loss = loss.item()
                    wait = 0
                else:
                    wait += 1
                if wait >= patience:
                    break
            self.optimized = True

    def forward(self, x):
        """
        Apply the neural adjoint map to the input tensor. Usually the optimization is performed on self.embedding2
        """
        if not self.optimized:
            raise Warning("Model not optimized. Please optimize before applying.")

        return self.model(x)
