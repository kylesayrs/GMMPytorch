import pytest
import torch

from src.utils import make_random_scale_trils, make_random_cov_matrix, warp_probs


@pytest.mark.parametrize(
    "num_sigmas,num_dims",
    [
        (1, 1),
        (5, 7),
        (10, 10),
    ],
)
def test_make_random_scale_trils(num_sigmas, num_dims):
    trils = make_random_scale_trils(num_sigmas, num_dims)
    assert trils.shape == torch.Size([num_sigmas, num_dims, num_dims])


@pytest.mark.parametrize(
    "num_dims,observations_per_variable",
    [
        (0, 10),
        (1, 10),
        (5, 10),
        (10, 10),
        (10, 2),
        (10, 1000),
    ],
)
def test_make_random_cov_matrix(num_dims, observations_per_variable):
    covariance = make_random_cov_matrix(num_dims, observations_per_variable)
    assert covariance.shape == torch.Size([num_dims, num_dims])


@pytest.mark.parametrize(
    "probs,target_value",
    [
        ([0.5, 0.5], 0.5),
        ([0.5, 0.5], 0.1),
        ([0.5, 0.5], 0.9),

        ([0.3, 1/3, 0.34], 0.5),
        ([0.3, 1/3, 0.34], 0.1),
        ([0.3, 1/3, 0.34], 0.9),

        ([0.1, 0.2, 0.3, 0.4], 0.5),
        ([0.1, 0.2, 0.3, 0.4], 0.1),
        ([0.1, 0.2, 0.3, 0.4], 0.9),
    ],
)
def test_warp_probs(probs, target_value):
    values = warp_probs(probs, target_value)

    target_value_domain = 1 / len(probs)
    for prob, value in zip(probs, values):
        if prob < target_value_domain:
            assert value < target_value

        if prob == target_value_domain:
            assert value == pytest.approx(target_value, 1e-5)

        if prob > target_value_domain:
            assert value > target_value
        
