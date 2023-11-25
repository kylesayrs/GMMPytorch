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


SIGMA_2D_0 = [
    [1, 0],
    [0, 1]
]

SIGMA_2D_1 = [
    [1, 0],
    [0, 1]
]

SIGMA_2D_2 = [
    [1, 0],
    [1, 1]
]


@pytest.mark.parametrize(
    "means,sigmas,exp_loss",
    [
        ([[ 0,  0]], [SIGMA_2D_0], 2.6),
        ([[ 1,  0]], [SIGMA_2D_0], 2.6),
        ([[ 0,  1]], [SIGMA_2D_0], 2.6),
        ([[-1,  0]], [SIGMA_2D_0], 2.6),
        ([[ 0, -1]], [SIGMA_2D_0], 2.6),

        ([[ 0,  0]], [SIGMA_2D_1], 2.6),
        ([[ 1,  0]], [SIGMA_2D_1], 2.6),
        ([[ 0,  1]], [SIGMA_2D_1], 2.6),
        ([[-1,  0]], [SIGMA_2D_1], 2.6),
        ([[ 0, -1]], [SIGMA_2D_1], 2.6),
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
