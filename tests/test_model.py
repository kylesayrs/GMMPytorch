import pytest
import torch
import numpy

from src.model import GmmFull
from src.fit_model import fit_model


SEED = 42
@pytest.fixture(scope="function")
def seed():
    numpy.random.seed(SEED)
    torch.manual_seed(SEED)


IDENTITY_2d = [
    [1, 0],
    [0, 1]
]
DIAG_LOW_VARIANCE_2d = [
    [0.5, 0],
    [0, 0.5]
]
DIAG_HIGH_VARIANCE_2d = [
    [10, 0],
    [0, 10]
]
@pytest.mark.parametrize(
    "means,sigmas,exp_loss",
    [
        ([[ 0,  0]], [IDENTITY_2d], 2.6),
        ([[ 1,  0]], [IDENTITY_2d], 2.6),
        ([[ 0,  1]], [IDENTITY_2d], 2.6),
        ([[-1,  0]], [IDENTITY_2d], 2.6),
        ([[ 0, -1]], [IDENTITY_2d], 2.6),

        ([[ 0,  0]], [DIAG_LOW_VARIANCE_2d], 1.9),
        ([[ 1,  0]], [DIAG_LOW_VARIANCE_2d], 1.9),
        ([[ 0,  1]], [DIAG_LOW_VARIANCE_2d], 1.9),
        ([[-1,  0]], [DIAG_LOW_VARIANCE_2d], 1.9),
        ([[ 0, -1]], [DIAG_LOW_VARIANCE_2d], 1.9),

        ([[ 0,  0]], [DIAG_HIGH_VARIANCE_2d], 4.9),
        ([[ 1,  0]], [DIAG_HIGH_VARIANCE_2d], 4.9),
        ([[ 0,  1]], [DIAG_HIGH_VARIANCE_2d], 4.9),
        ([[-1,  0]], [DIAG_HIGH_VARIANCE_2d], 4.9),
        ([[ 0, -1]], [DIAG_HIGH_VARIANCE_2d], 4.9),
    ],
)
def test_GmmFull(means, sigmas, exp_loss, seed):
    samples_per_cluster = 50
    num_iterations = 1000
    mixture_lr = 0.001
    component_lr = 0.01
    num_components = len(sigmas)
    num_dims = len(sigmas[0])
    
    data = numpy.concatenate([
        numpy.random.multivariate_normal(mean, sigma, samples_per_cluster)
        for mean, sigma in zip(means, sigmas)
    ])
    data = torch.tensor(data)

    model = GmmFull(num_components, num_dims, radius=numpy.max(means, axis=None))

    loss = fit_model(
        model,
        data,
        num_iterations,
        mixture_lr,
        component_lr,
        visualize=False
    )

    assert loss == pytest.approx(exp_loss, 0.1)
    #assert model.components.mu == means + pytest.approx(0.3)
