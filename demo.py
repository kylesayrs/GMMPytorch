import torch
import numpy
import argparse
import warnings

from data import sample_data
from model import get_model
from visualize import log_and_visualize
from FamilyTypes import FAMILY_NAMES, get_mixture_family_from_str


parser = argparse.ArgumentParser()
parser.add_argument("--samples", default=500)
parser.add_argument("--clusters", default=2)
parser.add_argument("--mixtures", default=5)
parser.add_argument("--dims", default=2)
parser.add_argument("--iterations", default=5_000)
parser.add_argument("--family", type=str, default="diagonal", choices=FAMILY_NAMES)
parser.add_argument("--log_freq", type=int, default=1_000)
parser.add_argument("--seed", type=int, default=43)


if __name__ == "__main__":
    args = parser.parse_args()
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.dims != 2:
        warnings.warn("Visualization is only supported for dims=2")
    
    # use mixture family for both generation and modelling
    mixture_family = get_mixture_family_from_str(args.family)

    # load data
    data, true_mus, true_sigmas = sample_data(
        args.samples, args.clusters, args.dims, mixture_family
    )

    # set up model
    model = get_model(mixture_family, args.mixtures, args.dims)

    # separate optimizers for mixture coeficients and components
    components_optimizer = torch.optim.Adam(model.component_parameters(), lr=1e-2)
    mixture_optimizer = torch.optim.Adam(model.mixture_parameters(), lr=1e-2)
    components_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(components_optimizer, 10_000)
    mixture_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(mixture_optimizer, 10_000)

    for batch_index in range(args.iterations):
        # zero grad
        components_optimizer.zero_grad()
        mixture_optimizer.zero_grad()

        # forward
        loss = model(data)

        # log and visualize
        if batch_index % args.log_freq == 0:
            log_and_visualize(data, model, loss)

        # backwards
        loss.backward()
        components_optimizer.step()
        mixture_optimizer.step()
        components_scheduler.step()
        mixture_scheduler.step()
    
    log_and_visualize(data, model, loss)
