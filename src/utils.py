import torch
import numpy
from sklearn.datasets import make_spd_matrix


def make_random_scale_trils(num_sigmas: int, num_dims: int) -> torch.Tensor:    
    return torch.tensor(numpy.array([
        numpy.tril(make_random_cov_matrix(num_dims))
        for _ in range(num_sigmas)
    ]))


def make_random_cov_matrix(num_dims: int, samples_per_variable: int = 10) -> numpy.ndarray:
    observations = numpy.random.normal(0, 1, (num_dims, samples_per_variable))
    return numpy.corrcoef(observations)


def trils_to_symmetric(trils: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        upper_i, upper_j = torch.triu_indices(*trils.shape[1:]).tolist()
        lower_i, lower_j = torch.tril_indices(*trils.shape[1:]).tolist()

        covariance_matrix = trils.clone()
        covariance_matrix[:, upper_i, upper_j] = trils[:, lower_i, lower_j]

        return covariance_matrix
