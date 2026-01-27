from functools import partial
import jax
import jax.numpy as jnp
from netket.utils import timing
from netket.stats.mc_stats import statistics, Stats

import sys
from pathlib import Path
current_path = Path(__file__).resolve()
project_root = None
for parent in current_path.parents:
    if (parent / ".project_root").exists():
        project_root = parent
        break
if project_root is None:
    raise RuntimeError("Could not find the project root. Make sure a '.project_root' file exists.")
sys.path.append(str(project_root))

from Custom_nk.expect import _chunk_local_sum, _batch_local_sum

@jax.jit
def custom_statistics(data):
    mean = jnp.mean(data).real.astype(jnp.float64)
    variance = jnp.var(data).real.astype(jnp.float64)
    error_of_mean = jnp.sqrt(variance / data.size)
    return Stats(mean, error_of_mean, variance)

@timing.timed
def expect_and_grad_lp(O, sigmas, apply_fun, variables, apply_fun_delta, variables_delta, O_loc_past=None, chunk_size=None, jax_operator=False, which="e"):
    if not jax_operator:
        O = O.to_jax_operator()
    if which == "g":
        return _lp_expect_and_grad_jax(O, sigmas, apply_fun_delta, variables_delta, apply_fun, variables, chunk_size, O_loc_past)
    elif which == "e":
        return _lp_expect_and_grad_jax(O, sigmas, apply_fun, variables, apply_fun_delta, variables_delta, chunk_size, O_loc_past)
    else:
        raise ValueError("Which has to be 'g' or 'e'!")

@partial(jax.jit, static_argnames=["apply_fun_g", "apply_fun_e", "chunk_size"])
def _lp_expect_and_grad_jax(O_jax, sigmas, apply_fun_g, variables_g, apply_fun_e, variables_e, chunk_size, O_loc_past):
    sigmas = sigmas.reshape((-1, sigmas.shape[-1]))
    n_samples = sigmas.shape[0]

    if chunk_size is not None:
        if n_samples % chunk_size != 0:
            raise ValueError("The chunk size must divede the number of samples!")
        sigmas = sigmas.reshape((n_samples // chunk_size, chunk_size, -1))
        local_sum = _chunk_local_sum(O_jax, sigmas, apply_fun_e, variables_e)
    else: 
        local_sum = _batch_local_sum(O_jax, sigmas, apply_fun_e, variables_e)  

    if O_loc_past is not None:
        std = jnp.sqrt(O_loc_past.variance)
        mean = O_loc_past.mean

    O_loc = statistics(local_sum)
    delta = 2 * (local_sum - O_loc.mean).conj() / n_samples

    grad_dtype = jax.tree_util.tree_leaves(variables_g)[0].dtype
    _, vjp_fun = jax.vjp(lambda x: apply_fun_g(x, sigmas), variables_g)
    grad =  jax.tree_util.tree_map(jnp.conj, vjp_fun(delta.astype(grad_dtype))[0]["params"])
    
    return O_loc, grad

@timing.timed
def expect_and_grad(O, sigmas, apply_fun, variables, O_loc_past=None, chunk_size=None, jax_operator=False):
    if not jax_operator:
        O = O.to_jax_operator()
    return _expect_and_grad_jax(O, sigmas, apply_fun, variables, chunk_size, O_loc_past)

def _expect_and_grad_jax(O_jax, sigmas, apply_fun, variables, chunk_size, O_loc_past):
    return _lp_expect_and_grad_jax(O_jax, sigmas, apply_fun, variables, apply_fun, variables, chunk_size, O_loc_past)

def clip_local_sum(local_sum, mean_past, std_past):
    return jnp.clip(local_sum, min=(mean_past - 5 * std_past), max=(mean_past + 5 * std_past))

@jax.jit
def _get_gradient_noise(local_sum, S, n_samples):
    delta = local_sum.reshape(-1, 1) - local_sum.mean()
    OEdata = 2 * S.conj() * delta * jnp.sqrt(n_samples)
    OE_var = OEdata.var(axis=0)
    OE_max = OEdata.max(axis=0)
    grad_error = jnp.sqrt(OE_var)

    return jnp.mean(grad_error) / jnp.sqrt(n_samples), jnp.mean(OE_max)