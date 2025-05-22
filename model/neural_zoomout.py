from .neural_adjoint_map import NeuralAdjointMap
from geomfum.refine import Refiner
from sklearn.neighbors import NearestNeighbors


class NeuralZoomOut(Refiner):
    """Neural Zoomout
    At each iteration, it computes a pointwise map,
    converts it back to a neural adjoint map.

    Parameters
    ----------
    nit : int
        Number of iterations.
    step : int or tuple[2, int]
        How much to increase each basis per iteration.
    atol : float
        Convergence tolerance.
        Ignored if step different than 1.
    """

    def __init__(
        self,
        nit=10,
        step=0,
        atol=None,
        p2p_from_fm_converter=None,
        fm_from_p2p_converter=None,
        iter_refiner=None,
    ):
        super().__init__()

        self.nit = nit
        self.step = step
        self.atol = atol

        if self._step_a != self._step_b != 0 and atol is not None:
            raise ValueError("`atol` can't be used with step different than 0.")

    @property
    def step(self):
        """How much to increase each basis per iteration.

        Returns
        -------
        step : tuple[2, int]
            Step.
        """
        return self._step_a, self._step_b

    @step.setter
    def step(self, step):
        """Set step.

        Parameters
        ----------
        step : int or tuple[2, int]
            How much to increase each basis per iteration.
        """
        if isinstance(step, int):
            self._step_a = self._step_b = step
        else:
            self._step_a, self._step_b = step

    def iter(self, nam, embedding1, embedding2):
        """Refiner iteration.

        Parameters
        ----------
        nam : NeuralAdjointMap
            Neural adjoint map.
        embedding1 : torch.Tensor
            First embedding.
        embedding2 : torch.Tensor
            Second embedding.
        Returns
        -------
        nam : NeuralAdjointMap
            Neural adjoint map.
        """
        k2, k1 = nam.embedding1.shape[-1], nam.embedding2.shape[-1]

        emb2_nn = nam(embedding2[:, :k2])
        knn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(
            embedding1[:, :k1].cpu().numpy()
        )
        distances, indices = knn.kneighbors(emb2_nn.detach().cpu().numpy())

        p2p = indices.flatten()

        nam = NeuralAdjointMap(embedding1, embedding2)

        nam.optimize_from_p2p(p2p)

        return nam

    def __call__(self, nam, embedding1, embedding2):
        """Apply refiner.

        Parameters
        ----------
        nam : NeuralAdjointMap
            Neural adjoint map.
        embedding1 : torch.Tensor
            First embedding.
        embedding2 : torch.Tensor
            Second embedding.
        Returns
        -------
        nam : NeuralAdjointMap
            refined Neural adjoint map.
        """
        k2, k1 = nam.embedding1.shape[-1], nam.embedding2.shape[-1]

        nit = self.nit
        for it in range(nit):
            new_nam = self.iter(
                nam,
                embedding1[:, : k1 + it * self.step[0]],
                embedding2[:, : k2 + it * self.step[1]],
            )
            nam = new_nam

        return nam
