import jax
import jax.numpy as jnp
from functools import partial
import time
import re
from typing import Dict

def change_par_dtype(par, dtype):
    def _change(x):
        if isinstance(x, jnp.ndarray):
            return x.astype(dtype)
        return x
    
    return jax.tree_util.tree_map(_change, par)

@partial(jax.jit, static_argnames=["dtype"])
def jitted_change_par_dtype(par, dtype):
    return jax.tree_util.tree_map(lambda x: x.astype(dtype), par)
    
def RE(x, x_true):
    return jnp.abs(x - x_true) / jnp.abs(x_true)

def get_params_string(params):
    params_str = ""
    if params is not None: 
        for key, value in params.items():
            params_str += f"_{key}{value}"
    
    return params_str
    
@jax.jit
def _spin_to_number(S):
    powers = 2 ** jnp.arange(S.shape[-1])[::-1]
    return jnp.dot(S, powers).astype(jnp.int64)

def spin_to_number(S):
    if S.dtype != jnp.int64:
        S = S.astype(jnp.int64)
    
    if S.ndim > 1:
        return jax.vmap(_spin_to_number)(S)
    else:
        return _spin_to_number(S)
    
def number_to_spin(n, L):
    return ((jnp.array([int(b) for b in format(n, f'0{L}b')]) * 2) - 1)
    
def get_PDF(S, L):
    N = int(2 ** L)
    PDF = jnp.bincount(S, minlength=N)
    return PDF / S.size

@jax.jit
def freq_KL(PDF_1, PDF_2, samples):
    eps = 1e-10
    indices = samples
    log_ratios = jnp.log(jnp.take(PDF_1, indices) + eps) - jnp.log(jnp.take(PDF_2, indices) + eps)
    return jnp.mean(log_ratios)

@jax.jit
def FE_KL(PDF, log_PDF, norm, PDF_2): 
    eps = 1e-10
    return jnp.dot(PDF / norm, log_PDF - jnp.log(PDF_2 + eps) - jnp.log(norm + eps))

@partial(jax.jit, static_argnames=["apply_fun_1"])
def MC_KL(apply_fun_1, variables_1, PDF_2, samples):
    eps = 1e-10
    indices = spin_to_number(samples)
    log_ratios = apply_fun_1(variables_1, samples) - jnp.log(jnp.take(PDF_2, indices) + eps)
    return jnp.mean(log_ratios)
    
def compute_RE(ref, x):
    return jnp.abs(ref - x) / jnp.abs(ref)

@partial(jax.jit, static_argnames = ["apply_fun"])
def model_to_PDF(apply_fun, sigmas, variables):
    log_PDF = 2 * apply_fun(variables, sigmas).real
    PDF = jnp.exp(log_PDF)
    norm = jnp.sum(PDF)

    return log_PDF, PDF, norm

@jax.jit
def FE_both_KL(PDF, log_PDF, norm, log_PDF_2, norm_2):
    eps = 1e-10
    return jnp.dot(PDF / norm, log_PDF - log_PDF_2 + jnp.log(norm_2 + eps) - jnp.log(norm + eps))

def time_sampling(sample_fun, n_samples, reps = 100):
    jax.block_until_ready(sample_fun(n_samples=n_samples))
    T = []
    
    for _ in range(reps):
        t0 = time.time()
        samples = sample_fun(n_samples=n_samples)
        jax.block_until_ready(samples)
        T.append(time.time() - t0)
    T = jnp.array(T)

    return jnp.mean(T), jnp.std(T)

def create_bins(x):
    return 0.5 * (x[::2] + x[1::2])

def binning_analysis(samples):
    l = len(samples)
    error = [jnp.std(samples) / jnp.sqrt(l)]
    while l > 16:
        samples = create_bins(samples)
        l = len(samples)
        error.append(jnp.std(samples) / jnp.sqrt(l))

    return error

def get_timing_info(file_path: str) -> Dict[str, float]:

    timing_data = {}
    total_pattern = re.compile(r"Total:\s*([0-9.]+)")

    try:
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip().startswith('#'):
                    break

                total_match = total_pattern.search(line)
                if total_match:
                    timing_data['Total'] = float(total_match.group(1))
                    continue

                if '|' in line and ':' in line:
                    try:
                        main_part = line.split('|')[-1] 
                        name_str, time_str_full = main_part.split(':')
                        name = name_str.strip()
                        time_val = float(time_str_full.strip().split(' ')[0])
                        
                        timing_data[name] = time_val
                    except (ValueError, IndexError):
                        continue

        
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}
    
    return timing_data