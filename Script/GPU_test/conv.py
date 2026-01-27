import jax 
import jax.numpy as jnp
import time
import numpy as np
import flax.linen as nn
from flax.nnx.nn import initializers

import sys
import os
sys.path.append(os.path.abspath('../..'))
from Archs.res_cnn import ResCNN

L = 100
filters = 32
kernel_shape = (3, 3)
n_res_blocks = 4
seed = 123

reps = 1
n_samples = 2**17
n_chains = 2**10

dtypes = [jnp.float64, jnp.float32, jnp.float16, jnp.bfloat16]
labels = ["f64", "f32", "f16", "bf16"]

for dtype, label in zip(dtypes, labels):
    print(f"\n=== dtype: {label} ===")
    x = jax.random.normal(jax.random.key(seed), (n_chains, n_samples // n_chains, L * L), dtype=dtype).astype(jnp.int8)
    layer = nn.Conv(features=filters, kernel_size=kernel_shape, use_bias=True, param_dtype=dtype, dtype=dtype,kernel_init=initializers.he_normal())
    key = jax.random.PRNGKey(seed)
    params = layer.init(key, x)

    apply_fun = lambda x: layer.apply(params, x)
    apply_fun = jax.jit(apply_fun)
    jax.block_until_ready(apply_fun(x))

    times = []
    for _ in range(reps):
        start = time.time()
        y = jax.block_until_ready(apply_fun(x))
        end = time.time()
        times.append((end - start))

    mean_t = np.mean(times)
    if label == "f64":
        t_ref = mean_t

    speedup = t_ref / mean_t
    print(f"time: {mean_t:.4f} s, speed-up vs f64: {speedup:.1f}x")