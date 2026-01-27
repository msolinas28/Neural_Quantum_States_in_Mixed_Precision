import os
os.environ["JAX_ENABLE_X64"] = "1"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import time
import numpy as np

key = jax.random.PRNGKey(0)
reps = 1           
matmuls_per_rep = 5
n = 15000     
batch = 5      

dtypes = [jnp.float64, jnp.float32, jnp.float16, jnp.bfloat16]
labels = ["f64", "f32", "f16", "bf16"]

print("Using devices:", jax.devices())

for label, dtype in zip(labels, dtypes):
    print(f"\n=== dtype: {label} ===")
    
    A = jax.random.normal(key, (batch, n, n), dtype=dtype) 
    B = jax.random.normal(key, (batch, n, n), dtype=dtype)
    if label == 'f64':
        print("After allocation:", jax.devices()[0].memory_stats())

    if dtype == jnp.float64:
        precision = jax.lax.Precision.HIGHEST
    else:
        precision = jax.lax.Precision.DEFAULT
    
    matmul = jax.jit(lambda X, Y: jnp.matmul(X, Y))
    
    matmul(A, B).block_until_ready()
    
    times = []
    for _ in range(reps):
        start = time.time()
        for _ in range(matmuls_per_rep):
            C = matmul(A, B).block_until_ready()
        end = time.time()
        times.append((end - start) / matmuls_per_rep) 
    
    mean_t = np.mean(times)
    if label == "f64":
        t_ref = mean_t
    
    speedup = t_ref / mean_t
    print(f"Matrix size: {n}x{n}, batch: {batch}, time: {mean_t:.4f} s, speed-up vs f64: {speedup:.1f}x")
