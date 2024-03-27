from typing import Iterator, List
from abc import ABC, abstractmethod

import torch
from torch.distributions.utils import logits_to_probs


class MixtureModel(ABC, torch.nn.Module):
    """
    Base model for mixture models

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
        super().__init__()
        self.num_components = num_components
        self.num_dims = num_dims

        self.logits = torch.nn.Parameter(torch.zeros(num_components, ))


    def mixture_parameters(self) -> Iterator[torch.nn.Parameter]:
        return iter([self.logits])


    def get_probs(self) -> torch.Tensor:
        return logits_to_probs(self.logits)
    

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    

    @abstractmethod
    def constrain_parameters(self):
        raise NotImplementedError()


    @abstractmethod
    def component_parameters(self) -> Iterator[torch.nn.Parameter]:
        raise NotImplementedError()
    

    @abstractmethod
    def get_covariance_matrix(self) -> torch.Tensor:
        raise NotImplementedError()
