import netket as nk
import argparse
import time
import random
import numpy as np
import jax
import tqdm

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

from mixed_rbm import RBM
from functions import RE
from hamiltonian_class import System
from project_paths import PROJECT_ROOT
from parse_utils import str2bool, dict_or_none, none_or_float, none_or_int, print_parsed
from jax_dtype_utils import jax_dtype
from Custom_nk.LPD import LPS_VMC

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="TFIM", type=str)
parser.add_argument(
    "--model_params",
    default='{"J": 1, "h": 0.5}',
    type=dict_or_none,
    help='Dictionary as a string, e.g., \'{"J": 1, "h": 0.5}\' or None',
)
parser.add_argument("--sampling_dtype", default="f64", type=jax_dtype)
parser.add_argument("--L", default=10, type = int)
parser.add_argument("--n_dim", default=1, type = int)
parser.add_argument("--PBC", default=True, type=str2bool)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--n_samples", default=2**12, type=int)
parser.add_argument("--M", default=1000, type = int)
parser.add_argument("--total_sz", default=None, type = none_or_float)
parser.add_argument("--n_chains", default=2**9, type=int)
parser.add_argument("--chunk_size", default=None, type=none_or_int)
parser.add_argument("--sweep_size", default=1, type=int)
parser.add_argument("--n_discard_per_chain", default=5, type=int)
parser.add_argument("--alpha", default=1, type=int)
parser.add_argument("--n_reps", default=10, type=int)
args = parser.parse_args()

print_parsed(args)

g = nk.graph.Hypercube(length=args.L, n_dim=args.n_dim, pbc=args.PBC)
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes, total_sz=args.total_sz)

model = RBM(alpha=args.alpha)

s = System(args.model, args.model_params, hi, g)

t_ref = []
t_delta = []

drivers = [nk.driver.VMC, LPS_VMC]
seeds = random.sample(range(0, args.n_reps * 1000), args.n_reps)

for i in tqdm.tqdm(range(len(seeds))):
    sampler = nk.sampler.MetropolisSampler(hi, rule=s.sampler_rule, n_chains=args.n_chains, sweep_size=args.sweep_size * hi.size) 
    opt = nk.optimizer.Sgd(learning_rate = args.lr)
    vs = nk.vqs.MCState(sampler, model, n_samples = args.n_samples, chunk_size = args.chunk_size, n_discard_per_chain=args.n_discard_per_chain, seed=seeds[i], sampler_seed=seeds[i])
    
    gs = LPS_VMC(
        hamiltonian = s.Hamiltonian,
        optimizer = opt,
        preconditioner = nk.optimizer.SR(diag_shift = 1e-3, holomorphic = False),
        variational_state = vs,
        lp_dtype = args.sampling_dtype.dtype
    )

    gs.run(n_iter=1)
    
    t_0 = time.time()
    gs.run(n_iter=args.M)
    jax.block_until_ready(gs.energy)
    t_delta.append(time.time() - t_0)

    sampler = nk.sampler.MetropolisSampler(hi, rule=s.sampler_rule, n_chains=args.n_chains, sweep_size=args.sweep_size * hi.size) 
    opt = nk.optimizer.Sgd(learning_rate = args.lr)
    vs = nk.vqs.MCState(sampler, model, n_samples = args.n_samples, chunk_size = args.chunk_size, n_discard_per_chain=args.n_discard_per_chain, seed=seeds[i], sampler_seed=seeds[i])
    
    gs = nk.driver.VMC(
        hamiltonian = s.Hamiltonian,
        optimizer = opt,
        preconditioner = nk.optimizer.SR(diag_shift = 1e-3, holomorphic = False),
        variational_state = vs
    )
    
    gs.run(n_iter=1)

    t_0 = time.time()
    gs.run(n_iter=args.M)
    jax.block_until_ready(gs.energy)
    t_ref.append(time.time() - t_0)

t_ref = np.array(t_ref)
t_delta = np.array(t_delta)

print("-----------------------")
print("delta: ", args.sampling_dtype.label)
print(rf"Time ref: {np.mean(t_ref)} $\pm$ {np.std(t_ref)}")
print(rf"Time delta: {np.mean(t_delta)} $\pm$ {np.std(t_delta)}")