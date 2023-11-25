from typing import Iterator

import torch
from torch.distributions import (
    Normal,
    Categorical,
    Independent,
    MultivariateNormal,
    MixtureSameFamily
)
from torch.distributions.utils import logits_to_probs

from src.FamilyTypes import MixtureFamily
from src.utils import make_random_scale_trils


class MixtureModel(torch.nn.Module):
    def __init__(
        self,
        num_components: int,
        num_dims: int,
        init_radius: float = 1.0,
    ):
        super().__init__()
        self.num_components = num_components
        self.num_dims = num_dims
        self.init_radius = init_radius

        self.logits = torch.nn.Parameter(torch.zeros(num_components, ))


    def mixture_parameters(self) -> Iterator[torch.nn.Parameter]:
        return iter([self.logits])


    def get_probs(self) -> torch.Tensor:
        return logits_to_probs(self.logits)


class GmmFull(MixtureModel):
    def __init__(
        self,
        num_components: int,
        num_dims: int,
        init_radius: float = 1.0,
    ):
        super().__init__(num_components, num_dims, init_radius)

        self.mus = torch.nn.Parameter(torch.rand(num_components, num_dims).uniform_(-init_radius, init_radius))
        self.scale_tril = torch.nn.Parameter(make_random_scale_trils(num_components, num_dims))
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixture = Categorical(logits=self.logits)
        components = MultivariateNormal(self.mus, scale_tril=self.scale_tril)
        mixture_model = MixtureSameFamily(mixture, components)

        nll_loss = -1 * mixture_model.log_prob(x).mean()

        # detect singularity collapse and reset
        if nll_loss.isnan():
            with torch.no_grad():
                self.__init__(self.num_components, self.num_dims, self.init_radius)

            return self.forward(x)

        return nll_loss
    

    def component_parameters(self) -> Iterator[torch.nn.Parameter]:
        return iter([self.mus, self.scale_tril])
    
    
    def get_covariance_matrix(self) -> torch.Tensor:
        return self.scale_tril @ self.scale_tril.mT
    

class GmmDiagonal(MixtureModel):
    """
    Implements diagonal gaussian mixture model

    :param num_components: number of components
    """
    def __init__(
        self,
        num_components: int,
        num_dims: int,
        init_radius: float = 1.0,
    ):
        super().__init__(num_components, num_dims, init_radius)

        self.mus = torch.nn.Parameter(torch.FloatTensor(num_components, num_dims).uniform_(-init_radius, init_radius))
        self.sigmas_diag = torch.nn.Parameter(torch.rand(num_components, num_dims))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixture = Categorical(logits=self.logits)
        components = Independent(Normal(self.mus, self.sigmas_diag), 1)
        mixture_model = MixtureSameFamily(mixture, components)

        nll_loss = -1 * mixture_model.log_prob(x).mean()

        # detect singularity collapse and reset
        if nll_loss.isnan():
            with torch.no_grad():
                self.__init__(self.num_components, self.num_dims, self.init_radius)

            return self.forward(x)

        return nll_loss
    

    def component_parameters(self) -> Iterator[torch.nn.Parameter]:
        return iter([self.mus, self.sigmas_diag])
    

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
