from .base import *
from .full import *
from .diagonal import *
from .isotropic import *


def get_model(
    mixture_family: MixtureFamily,
    num_components: int,
    num_dims: int,
    radius: float
) -> torch.nn.Module:
    if mixture_family == MixtureFamily.FULL:
        return GmmFull(num_components, num_dims, radius)
    
    if mixture_family == MixtureFamily.DIAGONAL:
        return GmmDiagonal(num_components, num_dims, radius)
    
    if mixture_family == MixtureFamily.ISOTROPIC:
        return GmmIsotropic(num_components, num_dims, radius)
    
    raise NotImplementedError(
        f"Mixture family {mixture_family.value} not implemented yet"
    )
