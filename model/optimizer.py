import torch
from .losses import *
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler




class LossHandler:
    def __init__(self,w_mse=1, w_cd=0, w_arap=0, w_smooth=0, w_uni=0, field=None):
        self.w_cd = w_cd
        self.w_arap = w_arap
        self.w_smooth = w_smooth
        self.w_mse=w_mse
    def compute(self, pred, target):
        """
        Compute the total loss using different geometric loss terms.
        
        Args:
        - f_y: Transformed target shape (via neural map).
        - x: Source shape.
        
        Returns:
        - loss: The combined loss value.
        """
        loss = F.mse_loss(pred, target)
        return self.w_mse*loss


class NAMOptimizer:
    def __init__(self, model, loss_handler, n_iter=200, learning_rate=0.01, patience=10, 
                 optimizer_cls=optim.Adam, scheduler_type=None, scheduler_params=None):
        """
        A class to optimize the Neural Adjoint Map model with various losses and learning rate scheduling.

        Args:
        - model (nn.Module): The neural map model (e.g., Neural_Map).
        - loss_handler (LossHandler): A loss handler that computes the relevant losses.
        - n_iter (int): The number of iterations for optimization.
        - learning_rate (float): The learning rate for the optimizer.
        - patience (int): Early stopping patience; how many epochs to wait before stopping.
        - optimizer_cls (Optimizer): The optimizer class to use (default is Adam).
        - scheduler_type (str): Type of scheduler to use (e.g., 'StepLR', 'CosineAnnealingLR').
        - scheduler_params (dict): Parameters for the scheduler (e.g., step size, decay rate, etc.).
        """
        self.model = model
        self.loss_handler = loss_handler
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.patience = patience
        self.optimizer_cls = optimizer_cls
        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params

        # Initialize optimizer
        self.optimizer = self.optimizer_cls(self.model.parameters(), lr=self.learning_rate)

        # Initialize learning rate scheduler if specified
        self.scheduler = None
        if scheduler_type:
            self._initialize_scheduler()

        # For early stopping
        self.best_loss = float('inf')
        self.wait = 0

    def _initialize_scheduler(self):
        """
        Initialize the learning rate scheduler based on the given type and parameters.
        """
        if self.scheduler_type == 'StepLR':
            self.scheduler = lr_scheduler.StepLR(self.optimizer, **self.scheduler_params)
        elif self.scheduler_type == 'CosineAnnealingLR':
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, **self.scheduler_params)
        elif self.scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, **self.scheduler_params)
        #else:
            #print(f"Scheduler type {self.scheduler_type} not recognized. No scheduler used.")

    def optimize(self, x, y):
        """
        Optimize the model with respect to the input shapes and losses.

        Args:
        - x (Tensor): Source embedding.
        - y (Tensor): Target embedding.
        - p2p (Tensor): Point-to-point correspondence.
        - x_field, y_field, neig, L (optional): Additional arguments for specific loss functions.
        """
        for iter in range(self.n_iter):
            self.optimizer.zero_grad()

            # Forward pass
            f_y = self.model(y)

            # Compute loss using the loss handler
            loss = self.loss_handler.compute(f_y, x)

            # Backpropagate and update model parameters
            loss.backward()
            self.optimizer.step()

            # Step the scheduler (if using StepLR or CosineAnnealingLR)
            if self.scheduler and isinstance(self.scheduler, lr_scheduler.StepLR):
                self.scheduler.step()

            # Step the scheduler (if using ReduceLROnPlateau)
            if self.scheduler and isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(loss)

            # Early stopping logic
            if loss < self.best_loss:
                self.best_loss = loss
                self.wait = 0  # Reset wait counter
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    #print(f"Early stopping at iteration {iter + 1}")
                    break

            # Print loss every 10 iterations
            #if iter % 10 == 0:
                #print(f"Iteration {iter + 1}/{self.n_iter}, Loss: {loss.item()}")

        return self.model
