import os
import jax

jax.config.update("jax_enable_x64", True)
import netket as nk
import matplotlib.pyplot as plt
import flax
from functools import partial
import math
import pickle

from typing import Any

from jax import numpy as jnp
from flax import linen as nn
from jax.nn.initializers import normal

from netket.utils.types import NNInitFunc

import numpy as np

from netket.hilbert import AbstractHilbert, HomogeneousHilbert
from netket.utils.types import DType
from netket.operator import DiscreteJaxOperator

default_kernel_init = normal(stddev=0.1)
import argparse

class IndicatorZeroStateJax(DiscreteJaxOperator):
    """
    Jax-compatible indicator function.
    """

    def __init__(
            self,
            hilbert: AbstractHilbert,
            *,
            dtype: DType = np.float64,
    ):
        super().__init__(hilbert)

        if set(self.hilbert.shape) != {2}:
            raise ValueError(
                "IndicatorZeroStateJax only supports Hamiltonians with two local states"
            )

        # check that it is homogeneous, throw error if it's not
        if not isinstance(self.hilbert, HomogeneousHilbert):
            local_states = self.hilbert.states_at_index(0)
            if not all(
                    np.allclose(local_states, self.hilbert.states_at_index(i))
                    for i in range(self.hilbert.size)
            ):
                raise ValueError(
                    "Hilbert spaces with non homogeneous local_states are not "
                    "yet supported by PauliStrings."
                )
        self._initialized = False
        self._dtype = dtype

    @property
    def dtype(self) -> DType:
        """The dtype of the operator's matrix elements ⟨σ|Ô|σ'⟩."""
        return self._dtype

    def get_conn_padded(self, x):

        x_ids = self.hilbert.states_to_local_indices(x)
        mels = jax_get_zero_state_projector_mels(x, self.dtype)
        return jnp.expand_dims(x_ids, axis=-2), mels


@partial(jax.jit, static_argnums=(1,))
def jax_get_zero_state_projector_mels(x, dtype):
    n = x.shape[-1]
    target = jnp.ones([1] * (x.ndim - 1) + [n])
    return jnp.all(x == target, axis=-1, keepdims=True).astype(dtype)


class IndicatorEvenJax(DiscreteJaxOperator):
    """
    Jax-compatible indicator function.
    """

    def __init__(
            self,
            hilbert: AbstractHilbert,
            *,
            dtype: DType = np.float64,
    ):
        super().__init__(hilbert)

        if set(self.hilbert.shape) != {2}:
            raise ValueError(
                "IndicatorZeroStateJax only supports Hamiltonians with two local states"
            )

        # check that it is homogeneous, throw error if it's not
        if not isinstance(self.hilbert, HomogeneousHilbert):
            local_states = self.hilbert.states_at_index(0)
            if not all(
                    np.allclose(local_states, self.hilbert.states_at_index(i))
                    for i in range(self.hilbert.size)
            ):
                raise ValueError(
                    "Hilbert spaces with non homogeneous local_states are not "
                    "yet supported by PauliStrings."
                )
        self._initialized = False
        self._dtype = dtype

    @property
    def dtype(self) -> DType:
        """The dtype of the operator's matrix elements ⟨σ|Ô|σ'⟩."""
        return self._dtype

    def get_conn_padded(self, x):

        x_ids = self.hilbert.states_to_local_indices(x)
        mels = jax_get_even_projector_mels(x, self.dtype)
        return jnp.expand_dims(x_ids, axis=-2), mels


@partial(jax.jit, static_argnums=(1,))
def jax_get_even_projector_mels(x, dtype):
    even = jnp.prod(x, -1)
    return jax.lax.eq(even, -1).astype(dtype)


def log_cosh(x, dtype):
    """
    Logarithm of the hyperbolic cosine, implemented in a more stable way.
    """
    scalar_log2 = jnp.array(np.log(2.0), dtype=dtype)
    scalar_neg2 = jnp.array(-2.0, dtype=dtype)
    scalar_1 = jnp.array(1.0, dtype=dtype)

    sgn_x = scalar_neg2 * jnp.signbit(x.real).astype(dtype) + scalar_1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - scalar_log2


class RBM(nn.Module):
    r"""A restricted boltzman Machine, equivalent to a 2-layer FFNN with a
    nonlinear activation function in between.
    """
    hilbert: AbstractHilbert = None
    sigma: float = 0.
    param_dtype: Any = np.float64
    """The dtype of the weights."""
    alpha: float | int = 1
    """feature density. Number of features equal to alpha * input.shape[-1]"""
    use_hidden_bias: bool = True
    """if True uses a bias in the dense layer (hidden layer bias)."""
    use_visible_bias: bool = True
    """if True adds a bias to the input not passed through the nonlinear layer."""
    precision: Any = None
    """numerical precision of the computation see :class:`jax.lax.Precision` for details."""

    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    hidden_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the hidden bias."""
    visible_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the visible bias."""

    @nn.compact
    def __call__(self, x_in):
        rand_array = self.param(
            "random_array",
            lambda key, shape, dtype: jax.random.normal(key, shape, dtype) * self.sigma,
            (2 ** self.hilbert.size,),
            self.param_dtype,
        )

        x = nn.Dense(
            name="Dense",
            features=int(self.alpha * x_in.shape[-1]),
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=self.use_hidden_bias,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
        )(x_in)

        x = log_cosh(x, self.param_dtype)
        x = jnp.sum(x, axis=-1, dtype=self.param_dtype)

        if self.use_visible_bias:
            v_bias = self.param(
                "visible_bias",
                self.visible_bias_init,
                (x_in.shape[-1],),
                self.param_dtype,
            )
            out_bias = jnp.dot(x_in, v_bias)
            x = x + out_bias
        indices = self.hilbert.states_to_numbers(x_in)
        noise = rand_array[indices]
        return x + noise / 2  # Account for sqrt(p)

    @property
    def label(self):
        return f"RBM_alpha{self.alpha}"


def bound_KL(sigma: float) -> float:
    return sigma


def bound_avg(sigma: float) -> float:
    return 2 * (1 - math.exp(sigma ** 2) * math.erfc(sigma))


def expect(obs, logpsi, x):
    xp, mels = obs.get_conn_padded(x)
    x = jax.lax.collapse(x, 0, 2)
    xp = jax.lax.collapse(xp, 0, 2)
    mels = jax.lax.collapse(mels, 0, 2)

    def kernel(_x, _xp, _mel):
        _x = jnp.expand_dims(_x, axis=0)
        return jnp.sum(_mel * jnp.exp(logpsi(_xp) - logpsi(_x)))

    local_kernel = jax.vmap(kernel, in_axes=0)(x, xp, mels)
    return nk.stats.statistics(local_kernel)


def get_evs(n, seed, sigma, obs_name, n_samples):
    dtype_ref = jnp.float64
    hilbert = nk.hilbert.Spin(s=0.5, N=n)
    n_therm = 2 ** 16
    model_ref = RBM(hilbert=hilbert, param_dtype=dtype_ref, alpha=1)
    sampler_ref = nk.sampler.MetropolisSampler(hilbert, rule=nk.sampler.rules.LocalRule(), n_chains_per_rank=2 ** 12)
    mc_state_ref = nk.vqs.MCState(sampler_ref, model_ref, n_samples=n_samples, seed=seed)

    model_noisy = RBM(hilbert=hilbert, param_dtype=dtype_ref, alpha=1, sigma=sigma)
    sampler_noisy = nk.sampler.MetropolisSampler(hilbert, rule=nk.sampler.rules.LocalRule(), n_chains_per_rank=2 ** 12)
    mc_state_noisy = nk.vqs.MCState(sampler_noisy, model_noisy, n_samples=n_samples)
    random_array = mc_state_noisy.parameters["random_array"]
    params_noisy = mc_state_ref.parameters.copy()
    params_noisy = flax.traverse_util.flatten_dict(params_noisy, sep='/')
    params_noisy["random_array"] = random_array.copy()
    mc_state_noisy.parameters = flax.traverse_util.unflatten_dict(params_noisy, sep='/')
    if obs_name == "X":
        ps = ["I"] * n
        ps[0] = "X"
        ps = "".join(ps)
        observable = nk.operator.PauliStringsJax(hilbert, operators=[ps, ])
    elif obs_name == "zero":
        observable = IndicatorZeroStateJax(hilbert)
    elif obs_name == "even":
        observable = IndicatorEvenJax(hilbert)
    else:
        raise NotImplementedError
    log_psis_fn_ref = jax.vmap(lambda _x: mc_state_ref._apply_fun({"params": mc_state_ref.parameters}, _x))
    # Thermalize
    diffs = []
    stds = []
    therm_steps = list(range(10))
    for t_step in therm_steps:
        samples_ref = mc_state_ref.sample(n_samples=n_therm)
        samples_noisy = mc_state_noisy.sample(n_samples=n_therm)
    samples = mc_state_ref.sample(n_samples=n_samples)
    acc_ref = mc_state_ref.sampler_state.acceptance
    stats_ref = expect(observable, log_psis_fn_ref, samples)
    samples = mc_state_noisy.sample(n_samples=n_samples)
    stats_noisy = expect(observable, log_psis_fn_ref, samples)
    acc_noisy = mc_state_noisy.sampler_state.acceptance

    return acc_ref, acc_noisy, stats_ref.to_dict(), stats_noisy.to_dict()


def main(n, n_samples, n_seeds):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        'text.latex.preamble': r'\usepackage{amsfonts}',
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 18,
        "font.size": 18,
        "legend.fontsize": 12,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    })
    # obs_name = 'indicator'
    obs_names = ["X", "even"]
    for obs_name in obs_names:
        sigma_list = np.logspace(-6, 1, 20)
        seeds = [100 * i for i in range(n_seeds)]
        for i, sigma in enumerate(sigma_list):
            for j, seed in enumerate(seeds):
                if not os.path.exists(f"./data_evs/{n}/{seed}/{obs_name}"):
                    os.makedirs(f"./data_evs/{n}/{seed}/{obs_name}")
                if not os.path.exists(f"./data_evs/{n}/{seed}/{obs_name}/stats_noisy_{sigma:1.7f}.c"):
                    out = get_evs(n=n, seed=seed, sigma=sigma, obs_name=obs_name, n_samples=n_samples)
                    np.save(f"./data_evs/{n}/{seed}/{obs_name}/acc_ref_{sigma:1.7f}", out[0])
                    np.save(f"./data_evs/{n}/{seed}/{obs_name}/acc_noisy_{sigma:1.7f}", out[1])
                    with open(f"./data_evs/{n}/{seed}/{obs_name}/stats_ref_{sigma:1.7f}.c", 'wb') as f:
                        pickle.dump(out[2], f)
                    with open(f"./data_evs/{n}/{seed}/{obs_name}/stats_noisy_{sigma:1.7f}.c", 'wb') as f:
                        pickle.dump(out[3], f)
                else:
                    print(f"n = {n}, seed = {seed} exists, skipping...")
    ### FIGURE ###
    fig, axs_bias = plt.subplots(1, 1)
    axs = [axs_bias, ]
    fig.set_size_inches(6, 6)
    reds = plt.get_cmap("Reds")
    oranges = plt.get_cmap("Oranges")
    purples = plt.get_cmap("Purples")
    blues = plt.get_cmap("Blues")
    greens = plt.get_cmap("Greens")
    cmap_val = 0.75
    alpha_val = 0.2
    alpha_spacing = np.linspace(0.5, cmap_val, n_seeds)
    color_map = {"X": reds, "zero": oranges, "even": purples}
    sampling_error ={}
    label_map = {"even": "Even", "X": r"$X_1$"}
    for obs_name in obs_names:
        cmap = plt.get_cmap(color_map[obs_name])
        col = cmap(cmap_val)
        acceptances = np.zeros((2, len(sigma_list), len(seeds)))
        means = np.zeros((2, len(sigma_list), len(seeds)))
        variances = np.zeros((2, len(sigma_list), len(seeds)))
        mc_errors = np.zeros((2, len(sigma_list), len(seeds)))
        for i, sigma in enumerate(sigma_list):
            for j, seed in enumerate(seeds):
                acceptances[0, i, j] = np.load(f"./data_evs/{n}/{seed}/{obs_name}/acc_ref_{sigma:1.7f}.npy")
                acceptances[1, i, j] = np.load(f"./data_evs/{n}/{seed}/{obs_name}/acc_noisy_{sigma:1.7f}.npy")
                with open(f"./data_evs/{n}/{seed}/{obs_name}/stats_ref_{sigma:1.7f}.c", 'rb') as f:
                    stats_ref = pickle.load(f)
                with open(f"./data_evs/{n}/{seed}/{obs_name}/stats_noisy_{sigma:1.7f}.c", 'rb') as f:
                    stats_noisy = pickle.load(f)
                if obs_name == "zero":
                    print(stats_ref, stats_noisy)
                means[0, i, j] = stats_ref['Mean']
                means[1, i, j] = stats_noisy['Mean']
                mc_errors[0, i, j] = stats_ref['Sigma']
                mc_errors[1, i, j] = stats_noisy['Sigma']
                print(stats_noisy)
        variances[np.isclose(variances, 0.0)] = np.nan
        acceptances[np.isclose(acceptances, 0.0)] = np.nan

        for j, seed in enumerate(seeds):
            axs_bias.plot(sigma_list, np.abs(means[0, :, j] - means[1, :, j]) / np.abs(means[0, :, j]),
                          label=fr"{label_map[obs_name]}" if j == (n_seeds - 1) else None,
                          marker='.', markersize=10, linestyle='', alpha=alpha_spacing[j], color=col)
            sampling_error[obs_name] = np.nanmax(4 * np.sqrt(
                (mc_errors[0] ** 2 + mc_errors[1] ** 2) / np.abs(means[0, :, :])), axis=1)
    for o1, o2 in zip(obs_names[:-1], obs_names[1:]):
        cmap = plt.get_cmap(color_map[o1])
        col = cmap(cmap_val)
        axs_bias.plot([8e-7] + list(sigma_list) + [20],
                      [sampling_error[o1][0]] + list(sampling_error[o1]) + [sampling_error[o1][-1]],
                      color=col)
        axs_bias.fill_between([8e-7] + list(sigma_list) + [50],
                              [sampling_error[o1][0]] + list(sampling_error[o1]) + [sampling_error[o1][-1]],
                              [sampling_error[o2][0]] + list(sampling_error[o2]) + [sampling_error[o2][-1]],
                              alpha=alpha_val,
                              color=col)
    o1 = obs_names[-1]
    cmap = plt.get_cmap(color_map[o1])
    col = cmap(cmap_val)
    axs_bias.plot([8e-7] + list(sigma_list) + [20],
                  [sampling_error[o1][0]] + list(sampling_error[o1]) + [sampling_error[o1][-1]],
                  color=col)
    axs_bias.fill_between([8e-7] + list(sigma_list) + [50], 1e-6,
                          [sampling_error[o1][0]] + list(sampling_error[o1]) + [sampling_error[o1][-1]],
                          alpha=alpha_val,
                          color=col)
    axs_bias.plot(sigma_list, list(map(bound_KL, sigma_list)), label="KL Bound",
                  marker='.', markersize=10, color=blues(cmap_val))
    axs_bias.plot(sigma_list, list(map(bound_avg, sigma_list)), label="MCMC: AVG Bound",
                  marker='.', markersize=10, color=greens(cmap_val))

    axs_bias.set_yscale('log')
    axs_bias.set_ylim(1e-6, 20)
    axs_bias.set_xlim(8e-7, 20)
    axs_bias.legend()
    axs_bias.set_ylabel(r'$|\mathbb{E}[f(x)]_{x\sim\pi(x)} - \mathbb{E}[f(x)]_{x\sim\tilde\pi(x)}|/ |\mathbb{E}[f(x)]_{x\sim\pi(x)}|$')

    xticks = np.logspace(-6, 1, 8)
    for ax in axs:
        ax.grid()
        ax.set_xscale("log")
        ax.set_xlabel(r"$\sigma$")
        ax.set_xticks(xticks)
    fig.tight_layout()
    fig.savefig(f"ev_n_{n}.pdf")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("n", default=4, type=int)
    parser.add_argument("n_samples", default=2**14,type=int)
    parser.add_argument("n_seeds", default=1, type=int)
    args = parser.parse_args()

    main(n=args.n, n_samples=args.n_samples, n_seeds=args.n_seeds)