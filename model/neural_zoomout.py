"""Conversion between pointwise and functional maps."""

import torch
import torch.nn as nn
import geomfum.backend as xgs

from geomfum.convert import BaseFmFromP2pConverter, BaseNeighborFinder, BaseP2pFromFmConverter, NeighborFinder
from geomfum.refine import ZoomOut
from geomfum.neural_adjoint_map import NeuralAdjointMap




class GPUEuclideanNeighborFinder(BaseNeighborFinder, nn.Module):
    """GPU-based Euclidean neighbor finder.
    
    Finds exact nearest neighbors using Euclidean distance on GPU.
    Uses brute-force distance computation for exact results.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors.
    """

    def __init__(self, n_neighbors=1):
        BaseNeighborFinder.__init__(self, n_neighbors=n_neighbors)
        nn.Module.__init__(self)

    def __call__(self, X, Y):
        """Return indices of the points in `X` nearest to the points in `Y`.

        Parameters
        ----------
        X : array-like, shape=[n_points_x, n_features]
            Reference points.
        Y : array-like, shape=[n_points_y, n_features]
            Query points.

        Returns
        -------
        neigs : array-like, shape=[n_points_x, n_neighbors]
            Indices of the nearest neighbors in Y for each point in X.
        """
        return self.forward(X, Y)

    def forward(self, X, Y):
        """Find k nearest neighbors using exact Euclidean distance.

        Parameters
        ----------
        X : array-like, shape=[n_points_x, n_features]
            Reference points.
        Y : array-like, shape=[n_points_y, n_features]
            Query points.

        Returns
        -------
        neigs : array-like, shape=[n_points_x, n_neighbors]
            Indices of the nearest neighbors in Y for each point in X.
        """
        # Compute squared Euclidean distances
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
        X_norm_sq = torch.sum(X**2, dim=1, keepdim=True)  # [n_points_x, 1]
        Y_norm_sq = torch.sum(Y**2, dim=1, keepdim=False)  # [n_points_y]
        similarity = torch.mm(X, Y.T)  # [n_points_x, n_points_y]
        
        distances = X_norm_sq + Y_norm_sq - 2 * similarity
        
        if self.n_neighbors == 1:
            # For single neighbor, use argmin (faster than topk)
            indices = torch.argmin(distances, dim=-1, keepdim=True)
        else:
            # Use topk with smallest=True for minimum distances
            indices = torch.topk(distances, self.n_neighbors, dim=-1, 
                               largest=False, sorted=False)[1]
        
        return indices

    def distance_matrix(self, X, Y):
        """Compute full distance matrix between X and Y.

        Parameters
        ----------
        X : array-like, shape=[n_points_x, n_features]
            Reference points.
        Y : array-like, shape=[n_points_y, n_features]
            Query points.

        Returns
        -------
        distances : array-like, shape=[n_points_x, n_points_y]
            Euclidean distance matrix.
        """
        X_norm_sq = torch.sum(X**2, dim=1, keepdim=True)
        Y_norm_sq = torch.sum(Y**2, dim=1, keepdim=False)
        similarity = torch.mm(X, Y.T)
        
        distances = X_norm_sq + Y_norm_sq - 2 * similarity
        return torch.sqrt(torch.clamp(distances, min=0))  # Clamp for numerical stability


class NamFromP2pConverter(BaseFmFromP2pConverter):
    """Neural Adjoint Map from pointwise map using Neural Adjoint Maps (NAMs)."""

    def __init__(self, iter_max=200, patience=10, min_delta=1e-4, device="cpu"):
        """Initialize the converter.

        Parameters
        ----------
        iter_max : int, optional
            Maximum number of iterations for training the Neural Adjoint Map.
        patience : int, optional
            Number of iterations with no improvement after which training will be stopped.
        min_delta : float, optional
            Minimum change in the loss to qualify as an improvement.
        device : str, optional
            Device to use for the Neural Adjoint Map (e.g., 'cpu' or 'cuda').
        """
        self.iter_max = iter_max
        self.device = device
        self.min_delta = min_delta
        self.patience = patience

    def __call__(self, p2p, basis_a, basis_b, optimizer=None):
        """Convert point to point map.

        Parameters
        ----------
        p2p : array-like, shape=[n_vertices_b]
            Pointwise map.
        basis_a : Basis,
            Basis of the source shape.
        basis_b : Basis,
            Basis of the target shape.
        optimizer : torch.optim.Optimizer, optional
            Optimizer for training the Neural Adjoint Map.

        Returns
        -------
        nam: NeuralAdjointMap , shape=[spectrum_size_b, spectrum_size_a]
            Neural Adjoint Map model.
        """
        evects1_pb = xgs.to_torch(basis_a.vecs[p2p, :]).to(self.device).double()
        evects2 = xgs.to_torch(basis_b.vecs).to(self.device).double()
        nam = NeuralAdjointMap(
            input_dim=basis_a.spectrum_size,
            output_dim=basis_b.spectrum_size,
            device=self.device,
        ).double()

        if optimizer is None:
            optimizer = torch.optim.Adam(nam.parameters(), lr=0.01, weight_decay=1e-5)

        best_loss = float("inf")
        wait = 0

        for _ in range(self.iter_max):
            optimizer.zero_grad()

            pred = nam(evects1_pb)

            loss = torch.nn.functional.mse_loss(pred, evects2)
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss - self.min_delta:
                best_loss = loss.item()
                wait = 0
            else:
                wait += 1
            if wait >= self.patience:
                break

        return nam


class P2pFromNamConverter(BaseP2pFromFmConverter):
    """Pointwise map from Neural Adjoint Map (NAM).

    Parameters
    ----------
    neighbor_finder : NeighborFinder
        Nearest neighbor finder.
    """

    def __init__(self, neighbor_finder=None):
        if neighbor_finder is None:
            neighbor_finder = NeighborFinder(n_neighbors=1)
        if neighbor_finder.n_neighbors > 1:
            raise ValueError("Expects `n_neighors = 1`.")

        self.neighbor_finder = neighbor_finder

    def __call__(self, nam, basis_a, basis_b):
        """Convert neural adjoint map.

        Parameters
        ----------
        nam : NeuralAdjointMap, shape=[spectrum_size_b, spectrum_size_a]
            Nam model.
        basis_a : Basis,
            Basis of the source shape.
        basis_b : Basis,
            Basis of the target shape.

        Returns
        -------
        p2p : array-like, shape=[n_vertices_b]
            Pointwise map.
        """
        k2, k1 = nam.shape

        emb1 = nam((basis_a.full_vecs[:, :k2]).to(nam.device).double())
        emb2 = (basis_b.full_vecs[:, :k1]).to(nam.device).double()

        p2p = self.neighbor_finder(emb2.detach(), emb1.detach()).flatten()
        return p2p


class NeuralZoomOut(ZoomOut):
    """Neural zoomout algorithm.

    Parameters
    ----------
    nit : int
        Number of iterations.
    step : int or tuple[2, int]
        How much to increase each basis per iteration.

    References
    ----------
    .. [VOM2025] Giulio Viganò, Maks Ovsjanikov, Simone Melzi.
        "NAM: Neural Adjoint Maps for refining shape correspondences".
    """

    def __init__(
        self,
        nit=10,
        step=1,
        p2p_from_nam_converter=P2pFromNamConverter(),
        device="cpu",
        
    ):
        super().__init__(
            nit=nit,
            step=step,
            p2p_from_fm_converter=p2p_from_nam_converter,
            fm_from_p2p_converter=NamFromP2pConverter(device=device),
        )