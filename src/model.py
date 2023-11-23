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

from src.FamilyTypes import MixtureFamily


class GmmFull(torch.nn.Module):
    def __init__(self, num_mixtures: int, num_dims: int, width: int):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.num_dims = num_dims
        self.width = width

        self.mus = torch.nn.Parameter(torch.rand(num_mixtures, num_dims).uniform_(-width, width))
        init_cov_factor = torch.rand(num_mixtures, num_dims, num_dims)
        init_scale_tril = torch.linalg.cholesky(init_cov_factor @ init_cov_factor.transpose(-2, -1))
        self.scale_tril = torch.nn.Parameter(init_scale_tril)

        self.mixture = Categorical(logits=torch.rand(num_mixtures, ))
        self.components = MultivariateNormal(self.mus, scale_tril=self.scale_tril)
        self.mixture_model = MixtureSameFamily(self.mixture, self.components)

        # workaround, see https://github.com/pytorch/pytorch/issues/114417
        self.mixture.logits.requires_grad = True


    def forward(self, x: torch.Tensor):
        # detect singularity collapse and reset
        if torch.any(self.scale_tril.isnan()):
            self.__init__(self.num_mixtures, self.num_dims, self.width)
            warnings.warn("Encountered singularity, model has been reset")

        return -1 * self.mixture_model.log_prob(x).mean()
    

    def forward(self, x: torch.Tensor):
        nll_loss = -1 * self.mixture_model.log_prob(x).mean()

        # detect singularity collapse and reset
        if nll_loss.isnan():
            with torch.no_grad():
                #pass
                self.mixture.logits.uniform_(0, 1)
                self.mus.data.uniform_(-self.width, self.width)
                init_cov_factor = torch.rand(self.num_mixtures, self.num_dims, self.num_dims)
                init_scale_tril = torch.linalg.cholesky(init_cov_factor @ init_cov_factor.mT)
                self.scale_tril.data = init_scale_tril

            nll_loss = -1 * self.mixture_model.log_prob(x).mean()

        return nll_loss
    

    def component_parameters(self) -> Iterator[torch.nn.Parameter]:
        return iter([self.mus, self.scale_tril])
    

    def mixture_parameters(self) -> Iterator[torch.nn.Parameter]:
        return iter([self.mixture.logits])
    

    def get_probs(self):
        return self.mixture.probs
    
    
    def get_covariance_matrix(self) -> torch.Tensor:
        return self.scale_tril @ self.scale_tril.transpose(-2, -1)
    

class GmmDiagonal(torch.nn.Module):
    def __init__(self, num_mixtures: int, num_dims: int, width: int):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.num_dims = num_dims
        self.width = width

        self.mus = torch.nn.Parameter(torch.rand(num_mixtures, num_dims).uniform_(-width, width))
        self.sigmas_diag = torch.nn.Parameter(torch.rand(num_mixtures, num_dims))

        self.mixture = Categorical(logits=torch.rand(num_mixtures, ))
        self.components = Independent(Normal(self.mus, self.sigmas_diag), 1)
        self.mixture_model = MixtureSameFamily(self.mixture, self.components)

        # workaround, see https://github.com/pytorch/pytorch/issues/114417
        self.mixture.logits.requires_grad = True


    def forward(self, x: torch.Tensor):
        nll_loss = -1 * self.mixture_model.log_prob(x).mean()

        # detect singularity collapse and reset
        if nll_loss.isnan():
            with torch.no_grad():
                self.mixture.logits.uniform_(0, 1)
                self.mus.data.uniform_(-self.width, self.width)
                self.sigmas_diag.data.uniform_(0, 1)

            nll_loss = -1 * self.mixture_model.log_prob(x).mean()

        return nll_loss
    

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
