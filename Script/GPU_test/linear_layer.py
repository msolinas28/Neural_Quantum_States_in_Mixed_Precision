import os 
os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
import time

import sys
import os
sys.path.append(os.path.abspath('../..'))
from Archs.rbm import RBM

n_samples = 2**17
n_chains = n_samples // 4

in_dim = 100
out_dim = 1
alpha = 10
seed = 123
reps = 1

dtypes = [jnp.float64, jnp.float32, jnp.float16, jnp.bfloat16]
labels = ["f64", "f32", "f16", "bf16"]

for dtype, label in zip(dtypes, labels):
    print(f"\n=== dtype: {label} ===")

    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(key, (n_chains, n_samples // n_chains, in_dim), dtype=dtype)

    # dense = nn.Dense(features=out_dim, use_bias=True, param_dtype=dtype, dtype=dtype, kernel_init=initializers.he_normal())
    dense = RBM(param_dtype=dtype, alpha=alpha)

    key = jax.random.PRNGKey(seed)
    params = dense.init(key, x)
    apply_fun = jax.jit(lambda x: dense.apply(params, x))

    jax.block_until_ready(apply_fun(x))
    times = []
    for _ in range(reps):
        start = time.time()
        y = jax.block_until_ready(apply_fun(x))
        end = time.time()
        times.append(end - start)

    mean_t = float(np.mean(times))
    if label == "f64":
        t_ref = mean_t

    speedup = t_ref / mean_t
    print(f"time: {mean_t:.4f} s, speed-up vs f64: {speedup:.1f}x")