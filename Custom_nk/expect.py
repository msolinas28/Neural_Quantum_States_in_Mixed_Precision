from functools import partial
import jax
import jax.numpy as jnp
from netket.stats import statistics

@jax.jit
def upcast(x):
    if x.dtype in [jnp.float16, jnp.bfloat16]:
        return x.astype(jnp.float32)
    return x

@partial(jax.jit, static_argnames=["apply_fun"])
def _local_sum_single(O, sigma, apply_fun, variables):
    etas, O_elements = O.get_conn_padded(sigma)
    O_elements = O_elements.astype(jax.tree_util.tree_leaves(variables)[0].dtype)
    log_sigma = upcast(apply_fun(variables, sigma)) 
    log_etas = upcast(apply_fun(variables, etas))
    ratio = jnp.exp(log_etas - log_sigma)

    return jnp.sum(ratio * O_elements)

@partial(jax.jit, static_argnames=["apply_fun"])
def _batch_local_sum(O, sigmas, apply_fun, variables):
    return jax.vmap(lambda s: _local_sum_single(O, s, apply_fun, variables))(sigmas)

@partial(jax.jit, static_argnames=["apply_fun"])
def _chunk_local_sum(O, sigmas, apply_fun, variables):
    return jax.vmap(lambda x: _batch_local_sum(O, x, apply_fun, variables))(sigmas)

def get_local_sum(O_jax, sigmas, apply_fun, variables, chunk_size=None):
    if len(sigmas.shape) == 1:
        local_sum = _local_sum_single(O_jax, sigmas, apply_fun, variables)
    else:
        sigmas = sigmas.reshape((-1, sigmas.shape[-1]))
        if chunk_size is not None: 
            if sigmas.shape[0] % chunk_size != 0:
                raise ValueError("The chunk size must divede the number of samples!")
            sigmas = sigmas.reshape((sigmas.shape[0] // chunk_size, chunk_size, -1))
            local_sum = _chunk_local_sum(O_jax, sigmas, apply_fun, variables)
        else: 
            local_sum = _batch_local_sum(O_jax, sigmas, apply_fun, variables)

    return local_sum

def expect(O, sigmas, apply_fun, variables, chunk_size=None):
    O_jax = O.to_jax_operator()
    return expect_jax(O_jax, sigmas, apply_fun, variables, chunk_size)

@partial(jax.jit, static_argnames=["apply_fun", "chunk_size"])
def expect_jax(O, sigmas, apply_fun, variables, chunk_size=None):
    local_sum = get_local_sum(O, sigmas, apply_fun, variables, chunk_size)
    return statistics(local_sum)