import netket as nk
import jax.numpy as jnp
import tqdm
import argparse
import numpy as np
import os

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

from functions import time_sampling
from project_paths import PROJECT_ROOT
from parse_utils import str2bool, print_parsed, none_or_int
from Custom_nk.LPS import Custom_sampler
from Archs.arch_utils import get_arch

parser = argparse.ArgumentParser()
parser.add_argument("--L", default=4, type = int)
parser.add_argument("--n_dim", default=2, type = int)
parser.add_argument("--PBC", default=True, type=str2bool)
parser.add_argument("--n_reps", default=10, type=int)
parser.add_argument("--chunk_size", default=None, type=none_or_int)
args = parser.parse_args()

print_parsed(args)

out_folder = f"{PROJECT_ROOT}Data/LPSE/Speedup"

dtype_ref = jnp.float64
dtypes = [jnp.float64, jnp.float32, jnp.float16, jnp.bfloat16]

N_samples = np.array([2**9, 2**11, 2**13, 2**15, 2**17])
N_chains = N_samples // 4
n_filters = [2, 4, 8, 16, 32]

seed = 123

g = nk.graph.Hypercube(length=args.L, n_dim=args.n_dim, pbc=args.PBC)
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

speedup = []
time_tot = []

for filters in tqdm.tqdm(n_filters):
    for n_samples, n_chains in zip(N_samples, N_chains):
        for dtype in dtypes:
            model = get_arch("ResCNN", g, param_dtype=dtype, n_res_blocks=4, filters=filters, kernel_shape=(3, 3), upcast_sums=False)
            sampler = Custom_sampler(hi, rule=nk.sampler.rules.LocalRule(), n_chains=n_chains, dtype_ratio=dtype) 
            vs = nk.vqs.MCState(sampler=sampler, model=model, n_samples=n_samples, seed=seed, sampler_seed=seed, chunk_size=args.chunk_size)
            t_mean, t_err = time_sampling(vs.sample, n_samples, args.n_reps)

            if dtype == jnp.float64:
                time_ref = t_mean

            time_tot.append(t_mean)
            speedup.append(time_ref / t_mean)
time_tot = np.array(time_tot).reshape((len(n_filters), len(N_samples), len(dtypes))).transpose((2, 0, 1))
speedup = np.array(speedup).reshape((len(n_filters), len(N_samples), len(dtypes))).transpose((2, 0, 1))

os.makedirs(out_folder, exist_ok=True)
np.savez(f"{out_folder}/ResCNN_grid.npz", speedup=speedup, time_tot=time_tot, n_samples=N_samples, n_chains=N_chains)