import torch
import numpy
from sklearn.datasets import make_spd_matrix


def make_random_scale_trils(num_sigmas: int, num_dims: int) -> torch.Tensor:    
    return torch.tensor(numpy.array([
        numpy.tril(make_spd_matrix(num_dims))
        for _ in range(num_sigmas)
    ]))


def tril_to_symmetric(tril: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        if len(tril.shape) == 2:
            tril = tril.unsqueeze(0)

        upper_i, upper_j = torch.triu_indices(*tril.shape[1:]).tolist()
        lower_i, lower_j = torch.tril_indices(*tril.shape[1:]).tolist()

        covariance_matrix = tril.clone()
        covariance_matrix[:, upper_i, upper_j] = tril[:, lower_i, lower_j]

        return covariance_matrix
