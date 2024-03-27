from typing import Iterator, List

import torch
from torch.distributions import (
    Normal,
    Categorical,
    Independent,
    MultivariateNormal,
    MixtureSameFamily
)

from src.model.base import MixtureModel
from src.FamilyTypes import MixtureFamily
from src.utils import make_random_scale_trils


class GmmFull(MixtureModel):
    """
    Gaussian mixture model with full covariance matrix expression

    :param num_components: Number of component distributions
    :param num_dims: Number of dimensions being modeled
    :param init_radius: L1 radius within which each component mean should
        be initialized, defaults to 1.0
    :param init_mus: mean values to initialize model with, defaults to None
    """
    def __init__(
        self,
        num_components: int,
        num_dims: int,
        init_radius: float = 1.0,
        init_mus: List[List[float]] = None
    ):
        super().__init__(num_components, num_dims)

        self.mus = torch.nn.Parameter(
            torch.tensor(init_mus, dtype=torch.float32)
            if init_mus is not None
            else torch.rand(num_components, num_dims).uniform_(-init_radius, init_radius)
        )
        
        # lower triangle representation of (symmetric) covariance matrix
        self.scale_tril = torch.nn.Parameter(make_random_scale_trils(num_components, num_dims))
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixture = Categorical(logits=self.logits)
        components = MultivariateNormal(self.mus, scale_tril=self.scale_tril)
        mixture_model = MixtureSameFamily(mixture, components)

        nll_loss = -1 * mixture_model.log_prob(x).mean()

        return nll_loss
    
    
    def constrain_parameters(self, epsilon: float = 1e-6):
        with torch.no_grad():
            for tril in self.scale_tril:
                # cholesky decomposition requires positive diagonal
                tril.diagonal().abs_()

                # diagonal cannot be too small (singularity collapse)
                tril.diagonal().clamp_min_(epsilon)
            

    def component_parameters(self) -> Iterator[torch.nn.Parameter]:
        return iter([self.mus, self.scale_tril])
    
    
    def get_covariance_matrix(self) -> torch.Tensor:
        return self.scale_tril @ self.scale_tril.mT
