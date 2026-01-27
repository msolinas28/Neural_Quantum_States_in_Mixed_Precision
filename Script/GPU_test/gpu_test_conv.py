import os
os.environ["JAX_ENABLE_X64"] = "1"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import time
import numpy as np
from jax import config
config.update("jax_default_matmul_precision", "highest")

key = jax.random.PRNGKey(0)
reps = 1
convs_per_rep = 5

# Convolution parameters
batch = 2**9
in_channels = 3    
out_channels = 32 
H, W = 256, 256   
kernel_size = (3, 3)
stride = (1, 1)
padding = "SAME"

dtypes = [jnp.float64, jnp.float32, jnp.float16, jnp.bfloat16]
labels = ["f64", "f32", "f16", "bf16"]

print("Using devices:", jax.devices())

for label, dtype in zip(labels, dtypes):
    print(f"\n=== dtype: {label} ===")

    x = jax.random.normal(key, (batch, H, W, in_channels), dtype=dtype) 
    w = jax.random.normal(key, (kernel_size[0], kernel_size[1], in_channels, out_channels), dtype=dtype)

    if label == 'f64':
        print("After allocation:", jax.devices()[0].memory_stats())

    # if dtype == jnp.float64:
    #     precision = jax.lax.Precision.HIGHEST
    # else:
    #     precision = jax.lax.Precision.DEFAULT

    conv_fn = jax.jit(lambda X, W: jax.lax.conv_general_dilated(
        X, W,
        window_strides=stride,
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC")
    ))

    conv_fn(x, w).block_until_ready()

    
    times = []
    for _ in range(reps):
        start = time.time()
        for _ in range(convs_per_rep):
            y = conv_fn(x, w).block_until_ready()
        end = time.time()
        times.append((end - start) / convs_per_rep)

    mean_t = np.mean(times)
    if label == "f64":
        t_ref = mean_t

    speedup = t_ref / mean_t
    print(f"Image size: {H}x{W}, batch: {batch}, filters: {out_channels}, time: {mean_t:.4f} s, speed-up vs f64: {speedup:.1f}x")