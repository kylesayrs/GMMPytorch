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


def scale_probs(probs: torch.Tensor, uniform_part_value: float = 0.75) -> torch.Tensor:
    """
    (1/n)**a = 1/2
    a * log(1/n) = log(1/2)
    a = log(1/2) / log(1/n)

    :param probs: tensor describing the probability of each event
    :param uniform_part_value: the values a uniform distribution would be assigned
    :return: probs rescaled such that 1/len(probs) = 1/2
    """
    alpha = numpy.log(uniform_part_value) / numpy.log(1 / len(probs))
    return probs ** alpha
