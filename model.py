from typing import Iterator

import torch
from torch.distributions import (
    Categorical,
    MultivariateNormal,
    Independent,
    Normal,
    MixtureSameFamily
)

from FamilyTypes import MixtureFamily


class GmmFull(torch.nn.Module):
    def __init__(self, num_mixtures: int, num_dims: int, width: int):
        super().__init__()
        self.mus = torch.nn.Parameter(torch.rand(num_mixtures, num_dims) * width)
        self.sigmas_factor = torch.nn.Parameter(torch.rand(num_mixtures, num_dims, num_dims))

        self.mixture = Categorical(logits=torch.rand(num_mixtures, ))
        self.components = MultivariateNormal(self.mus, self.get_covariance_matrix())
        self.mixture_model = MixtureSameFamily(self.mixture, self.components)

        # workaround, see https://github.com/pytorch/pytorch/issues/114417
        self.mixture.logits.requires_grad = True
    

    def forward(self, x: torch.Tensor):
        return -1 * self.mixture_model.log_prob(x).mean()
    

    def component_parameters(self) -> Iterator[torch.nn.Parameter]:
        return iter([self.mus, self.sigmas_factor])
    

    def mixture_parameters(self) -> Iterator[torch.nn.Parameter]:
        return iter([self.mixture.logits])
    

    def get_probs(self):
        return self.mixture.probs
    
    
    def get_covariance_matrix(self) -> torch.Tensor:
        return self.sigmas_factor @ self.sigmas_factor.transpose(-2, -1)
    

class GmmDiagonal(torch.nn.Module):
    def __init__(self, num_mixtures: int, num_dims: int, width: int):
        super().__init__()
        self.mus = torch.nn.Parameter(torch.rand(num_mixtures, num_dims) * width)
        self.sigmas_diag = torch.nn.Parameter(torch.rand(num_mixtures, num_dims))

        self.mixture = Categorical(logits=torch.rand(num_mixtures, ))
        self.components = Independent(Normal(self.mus, self.sigmas_diag), 1)
        self.mixture_model = MixtureSameFamily(self.mixture, self.components)

        # workaround, see https://github.com/pytorch/pytorch/issues/114417
        self.mixture.logits.requires_grad = True


    def forward(self, x: torch.Tensor):
        return -1 * self.mixture_model.log_prob(x).mean()
    

    def component_parameters(self) -> Iterator[torch.nn.Parameter]:
        return iter([self.mus, self.sigmas_diag])
    

    def mixture_parameters(self) -> Iterator[torch.nn.Parameter]:
        return iter([self.mixture.logits])
    

    def get_probs(self):
        return self.mixture.probs

    
    def get_covariance_matrix(self) -> torch.Tensor:
        return torch.diag_embed(self.sigmas_diag)


def get_model(
    mixture_family: MixtureFamily,
    num_mixtures: int,
    num_dims: int,
    width: float
) -> torch.nn.Module:
    if mixture_family == MixtureFamily.FULL:
        return GmmFull(num_mixtures, num_dims, width)
    
    if mixture_family == MixtureFamily.DIAGONAL:
        return GmmDiagonal(num_mixtures, num_dims, width)
    
    raise NotImplementedError(
        f"Mixture family {mixture_family.value} not implemented yet"
    )
