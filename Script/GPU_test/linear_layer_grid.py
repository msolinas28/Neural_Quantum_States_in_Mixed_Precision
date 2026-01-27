import os 
os.environ["JAX_ENABLE_X64"] = "1"
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import time

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
from Archs.rbm import RBM

n_samples = np.array([2**9, 2**11, 2**13, 2**15, 2**17])
n_chains = n_samples // 4

in_dim = 100
out_dim = 1
alphas = [1, 2, 4, 6, 8, 10]
seed = 123
reps = 1

dtypes = [jnp.float64, jnp.float32, jnp.float16, jnp.bfloat16]
labels = ["f64", "f32", "f16", "bf16"]

speedup = []
elapsed_time = []
key = jax.random.PRNGKey(seed)

for alpha in tqdm.tqdm(alphas):
    for n_s, n_c in zip(n_samples, n_chains): 
        for dtype, label in zip(dtypes, labels):            
            x = jax.random.normal(key, (n_c, n_s // n_c, in_dim), dtype=dtype)
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
            speedup.append(t_ref / mean_t)
            elapsed_time.append(mean_t)

speedup = np.array(speedup).reshape((len(alphas), len(n_samples), len(dtypes))).transpose((2, 0, 1))
elapsed_time = np.array(elapsed_time).reshape((len(alphas), len(n_samples), len(dtypes))).transpose((2, 0, 1))

folder = f'{PROJECT_ROOT}Data/GPU_test'
os.makedirs(folder, exist_ok=True)
np.savez(f"{folder}/RBM.npz", speedup=speedup, elapsed_time=elapsed_time)