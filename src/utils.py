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
