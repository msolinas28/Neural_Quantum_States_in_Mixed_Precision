import os
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=4"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90" 
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
import time
import pynvml
import pandas as pd
from tqdm import tqdm

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

from project_paths import PROJECT_ROOT

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def gpu_mem_gb():
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1e9, info.total / 1e9

key = jax.random.PRNGKey(0)
reps = 50

Ns = [1000, 2000, 4000, 8000, 16000]
BATCHES = [1]

dtypes = [jnp.float64, jnp.float32, jnp.float16, jnp.bfloat16]
labels = ["f64", "f32", "f16", "bf16"]

results = []

for n in tqdm(Ns):
    for batch in BATCHES:
        times_ref = None

        for label, dtype in zip(labels, dtypes):
            used_before, total = gpu_mem_gb()
            try:
                A = jax.random.normal(key, (batch, n, n), dtype=dtype)
                B = jax.random.normal(key, (batch, n, n), dtype=dtype)

                matmul = jax.jit(lambda X, Y: jnp.matmul(X, Y))
                matmul(A, B).block_until_ready()

                times = []
                for _ in range(reps):
                    start = time.time()
                    C = matmul(A, B).block_until_ready()
                    end = time.time()
                    times.append((end - start))
                mean_t = np.mean(times)

                if label == "f64":
                    times_ref = mean_t

                speedup = times_ref / mean_t if times_ref else 1.0

                used_after, _ = gpu_mem_gb()
                results.append((n, batch, label, mean_t, speedup, used_after))

                del A, B, C

            except RuntimeError as e:
                print(f"OOM or error at n={n}, batch={batch}, dtype={label}: {e}")
                results.append((n, batch, label, np.nan, np.nan, np.nan))

df = pd.DataFrame(results, columns=["n", "batch", "dtype", "time_s", "speedup", "mem_used_GB"])
df.to_csv(f'{PROJECT_ROOT}Data/GPU_test/matmul.csv', index=False)