import torch
import numpy
from sklearn.datasets import make_spd_matrix


def make_random_scale_trils(num_sigmas: int, num_dims: int) -> torch.Tensor:
    """
    Make random lower triangle scale matrix. Generated by taking the The lower
    triangle of a random covariance matrix

    :param num_sigmas: number of matrices to make
    :param num_dims: covariance matrix size
    :return: random lower triangular scale matrices
    """
    return torch.tensor(numpy.array([
        numpy.tril(make_random_cov_matrix(num_dims))
        for _ in range(num_sigmas)
    ]))


def make_random_cov_matrix(num_dims: int, observations_per_variable: int = 10) -> numpy.ndarray:
    """
    Make random covariance matrix using observation sampling

    :param num_dims: number of variables described by covariance matrix
    :param samples_per_variable: number of observations for each variable used
        to generated covariance matrix
    :return: random covariance matrix
    """
    if num_dims == 1:
        return numpy.array([[1.0]])

    observations = numpy.random.normal(0, 1, (num_dims, observations_per_variable))
    return numpy.corrcoef(observations)


def warp_probs(probs: torch.Tensor, target_value: float = 0.75) -> torch.Tensor:
    """
    Warps probability distribution such that, for a list of probabilities of
    length n, the value 1/n becomes `target_value`.

    Derivation:
    (1/n) ** a = t
    a * log(1/n) = log(t)
    a = log(t) / log(1/n)

    :param probs: tensor describing the probability of each event
    :param target_value: the value 1/n would be assigned after scaling 
    :return: probs rescaled such that 1/len(probs) = 1/2
    """
    alpha = numpy.log(target_value) / numpy.log(1 / len(probs))
    return probs ** alpha
