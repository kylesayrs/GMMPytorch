from typing import Iterator

import torch
import warnings
from torch.distributions import (
    Normal,
    Categorical,
    Independent,
    MultivariateNormal,
    MixtureSameFamily
)
from sklearn.datasets import make_spd_matrix

from src.FamilyTypes import MixtureFamily
from src.utils import make_random_scale_trils


class GmmFull(torch.nn.Module):
    def __init__(
        self,
        num_components: int,
        num_dims: int,
        radius: float = 1.0,
        seed: int = 42
    ):
        torch.manual_seed(0)

        super().__init__()
        self.num_components = num_components
        self.num_dims = num_dims
        self.radius = radius
        self.seed = seed
        self._num_resets = 0

        self.mus = torch.nn.Parameter(torch.rand(num_components, num_dims).uniform_(-radius, radius))
        self.scale_tril = torch.nn.Parameter(make_random_scale_trils(num_components, num_dims))

        self.mixture = Categorical(logits=torch.zeros(num_components, ))
        self.components = MultivariateNormal(self.mus, scale_tril=self.scale_tril)
        self.mixture_model = MixtureSameFamily(self.mixture, self.components)

        # workaround, see https://github.com/pytorch/pytorch/issues/114417
        self.mixture.logits.requires_grad = True
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nll_loss = -1 * self.mixture_model.log_prob(x).mean()

        # detect singularity collapse and reset
        if nll_loss.isnan():
            _num_resets += 1
            with torch.no_grad():
                self.mixture.logits.uniform_(0, 1)
                self.mus.data.uniform_(-self.radius, self.radius)
                self.scale_tril.data = make_random_scale_trils(self.num_components, self.num_dims)

            nll_loss = -1 * self.mixture_model.log_prob(x).mean()

        return nll_loss
    

    def component_parameters(self) -> Iterator[torch.nn.Parameter]:
        return iter([self.mus, self.scale_tril])
    

    def mixture_parameters(self) -> Iterator[torch.nn.Parameter]:
        return iter([self.mixture.logits])
    

    def get_probs(self) -> torch.Tensor:
        return self.mixture.probs
    
    
    def get_covariance_matrix(self) -> torch.Tensor:
        return self.scale_tril @ self.scale_tril.mT
    

class GmmDiagonal(torch.nn.Module):
    """
    Implements digonal gaussian mixture model

    :param num_components: number of components
    """
    def __init__(
        self,
        num_components: int,
        num_dims: int,
        radius: float = 1.0,
        seed: int = 42,
    ):
        torch.manual_seed(0)

        super().__init__()
        self.num_components = num_components
        self.num_dims = num_dims
        self.radius = radius
        self.seed = seed
        self._num_resets = 0

        self.mus = torch.nn.Parameter(torch.FloatTensor(num_components, num_dims).uniform_(-radius, radius))
        self.sigmas_diag = torch.nn.Parameter(torch.rand(num_components, num_dims))

        self.mixture = Categorical(logits=torch.zeros(num_components, ))
        self.components = Independent(Normal(self.mus, self.sigmas_diag), 1)
        self.mixture_model = MixtureSameFamily(self.mixture, self.components)

        # workaround, see https://github.com/pytorch/pytorch/issues/114417
        self.mixture.logits.requires_grad = True


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nll_loss = -1 * self.mixture_model.log_prob(x).mean()

        # detect singularity collapse and reset
        if nll_loss.isnan():
            _num_resets += 1
            with torch.no_grad():
                self.mixture.logits.uniform_(0, 1)
                self.mus.data.uniform_(-self.radius, self.radius)
                self.sigmas_diag.data.uniform_(0, 1)

            nll_loss = -1 * self.mixture_model.log_prob(x).mean()

        return nll_loss
    

    def component_parameters(self) -> Iterator[torch.nn.Parameter]:
        return iter([self.mus, self.sigmas_diag])
    

    def mixture_parameters(self) -> Iterator[torch.nn.Parameter]:
        return iter([self.mixture.logits])
    

    def get_probs(self) -> torch.Tensor:
        return self.mixture.probs

    
    def get_covariance_matrix(self) -> torch.Tensor:
        return torch.diag_embed(self.sigmas_diag)


def get_model(
    mixture_family: MixtureFamily,
    num_components: int,
    num_dims: int,
    radius: float
) -> torch.nn.Module:
    if mixture_family == MixtureFamily.FULL:
        return GmmFull(num_components, num_dims, radius)
    
    if mixture_family == MixtureFamily.DIAGONAL:
        return GmmDiagonal(num_components, num_dims, radius)
    
    raise NotImplementedError(
        f"Mixture family {mixture_family.value} not implemented yet"
    )
