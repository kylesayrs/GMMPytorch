# Gaussian Mixture Models in Pytorch #
Implements gaussian mixture models in pytorch. Loss is computed with respect to mean negative log likelihood and optimized via gradient descent. Can be stacked on top of a fully connected layer to create a mixture density network (MDN).

<p align="center">
<img width="75%" src="assets/5_clusters.png" alt="Example Optimization"/>
</p>

## Usage ##
Run demo
```
usage: demo.py [-h] [--samples SAMPLES] [--components COMPONENTS] [--dims DIMS]
               [--iterations ITERATIONS]
               [--family {full,diagonal,isotropic,shared_isotropic,constant}] [--log_freq LOG_FREQ]
               [--radius RADIUS] [--mixture_lr MIXTURE_LR] [--component_lr COMPONENT_LR]
               [--visualize VISUALIZE] [--seed SEED]

Fit a gaussian mixture model to generated mock data

options:
  -h, --help            show this help message and exit
  --samples SAMPLES     The number of total samples in dataset
  --components COMPONENTS
                        The number of gaussian components in mixture model
  --dims DIMS           The number of data dimensions
  --iterations ITERATIONS
                        The number optimization steps
  --family {full,diagonal,isotropic,shared_isotropic,constant}
                        Model family, see `Mixture Types`
  --log_freq LOG_FREQ   Steps per log event
  --radius RADIUS       L1 bound of data samples
  --mixture_lr MIXTURE_LR
                        Learning rate of mixture parameter (pi)
  --component_lr COMPONENT_LR
                        Learning rate of component parameters (mus, sigmas)
  --visualize VISUALIZE
                        True for visualization at each log event and end
  --seed SEED           seed for numpy and torch

```

Fit model
```python3
data = load_data(...)

model = GmmFull(num_components=3, num_dims=2)

fit_model(
    model,
    data,
    num_iterations=10_000,
    mixture_lr=1e-5
    component_lr=1e-2
)
```

Run tests
```bash
python3 -m pytest tests
```

## Derivation ##
We start with the probability density function of a multivariate gaussian parameterized by mean $\mu \in \mathbb{R}^{d}$ and the covariance matrix $\Sigma \in \mathrm{S}_+^d$. The PDF describes the likelihood of sampling a point $x\in\mathbb{R}^{d}$ from the distribution.

```math
\mathcal{N}(\mathbf{x}) = \frac{1}{(2\pi)^{k/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)
```

In order to describe a mixture of gaussians, we add an additional parameter $\pi_k \in \Delta^{k-1}$ which describes the probability that a sample comes from any of the $K gaussian components.

```math
p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
```

Given elements of a dataset $x \in X^{(D \times N)}$, we want our model to fit to the data. This means maximizing the likelihood that the elements could have been sampled from the mixture PDF $p(\mathbf{x})$. Applying the function $-log(p(\mathbf{x}))$ for each element $\mathbf{x}$ has the effect of lowerbounding the best possible probability (1) while leaving the cost of an unlikely point (~0) unbounded. Given a dataset which contains few outliers and sufficently many components to cover the dataset, these properties make the negative log likelihood a suitable choice for our objective function.

```math
f(\mathbf{x}) = - \frac{1}{N} \sum_{i=1}^{N} \ln{ \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) }
```

For a from-scratch implementation of negative log likelihood backpropogation, see [GMMScratch](https://github.com/kylesayrs/GMMScratch/tree/master)


## Gaussian Model Types ##
| Type       | Description                                                                   |
| ---------- | ----------------------------------------------------------------------------- |
| Full       | Fully expressive eigenvalues. Data can be skewed in any direction             |
| Diagonal   | Eigenvalues align with data axes. Dimensional variance is indepedent          |
| Isotropic  | Equal variance in all directions. Spherical distributions                     |
| Shared     | Equal variance in all directions for all components                           |
| Constant   | Variance is not learned and is equal across all dimensions and components     |

As of now only Full and Diagonal mixture types have been implemented

## Singularity Mitigation ##
From Pattern Recognition and Machine Learning by Christopher M. Bishop, pg. 433:
> Suppose that one of the components of the mixture model, let us say the jth component, has its mean μ_j exactly equal to one of the data points so that μ_j = x_n for some value of n. If we consider the limit σ_j → 0, then we see that this term goes to infinity and so the log likelihood function will also go to infinity. Thus the maximization of the log likelihood function is not a well posed problem because such singularities will always be present and will occur whenever one of the Gaussian components ‘collapses’ onto a specific data point.

A common solution to this problem is to reset the mean of the offending component whenever a singularity appears. In practice, singularities can be mitigated by clamping the minimum value of elements on the covariance diagonal. In a stochastic environment, a large enough clamp value will allow the model to recover after a few iterations.
