from typing import Iterator, List, Optional
from abc import ABC, abstractmethod

import torch
from torch.distributions.utils import logits_to_probs

from src.visualize import plot_data_and_model


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
    

    def fit(
        self,
        data: torch.Tensor,
        num_iterations: int,
        mixture_lr: float,
        component_lr: float,
        log_freq: Optional[int] = None,
        visualize: bool = True
    ) -> float:
        # create separate optimizers for mixture coeficients and components
        mixture_optimizer = torch.optim.Adam(self.mixture_parameters(), lr=mixture_lr)
        mixture_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(mixture_optimizer, num_iterations)
        components_optimizer = torch.optim.Adam(self.component_parameters(), lr=component_lr)
        components_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(components_optimizer, num_iterations)

        # optimize
        for iteration_index in range(num_iterations):
            # reset gradient
            components_optimizer.zero_grad()
            mixture_optimizer.zero_grad()

            # forward
            loss = self(data)

            # log and visualize
            if log_freq is not None and iteration_index % log_freq == 0:
                print(f"Iteration: {iteration_index:2d}, Loss: {loss.item():.2f}")
                if visualize:
                    plot_data_and_model(data, self)

            # backwards
            loss.backward()
            mixture_optimizer.step()
            mixture_scheduler.step()
            components_optimizer.step()
            components_scheduler.step()

            # constrain parameters
            self.constrain_parameters()

        return float(loss.detach())
    

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
