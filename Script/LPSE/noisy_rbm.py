import netket as nk
import jax.numpy as jnp
import tqdm
import argparse
import os
import pandas as pd
import random

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

from Archs.noisy_rbm import NoisyRBM
from Archs.rbm import RBM
from functions import get_params_string, compute_RE
from file_handler import VarHandler
from Custom_nk.expect import expect
from hamiltonian_class import System
from project_paths import PROJECT_ROOT
from parse_utils import str2bool, dict_or_none, none_or_float, print_parsed, none_or_int
from observables import total_magnetization

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="TFIM", type=str)
parser.add_argument(
    "--model_params",
    default='{"J": 1, "h": 0.5}',
    type=dict_or_none,
    help='Dictionary as a string, e.g., \'{"J": 1, "h": 0.5}\' or None',
)
parser.add_argument("--L", default=12, type = int)
parser.add_argument("--n_dim", default=1, type = int)
parser.add_argument("--PBC", default=True, type=str2bool)
parser.add_argument("--n_samples", default=2**17, type=int)
parser.add_argument("--total_sz", default=None, type = none_or_float)
parser.add_argument("--n_therm", default=2**12, type = int)
parser.add_argument("--n_chains", default=16, type=int)
parser.add_argument("--n_reps", default=5, type=int)
parser.add_argument("--alpha", default=1, type=int)
parser.add_argument("--chunk_size", default=None, type=none_or_int)
parser.add_argument("--sigma", default=1e-3, type=float)
args = parser.parse_args()

print_parsed(args)

out_folder = f"{PROJECT_ROOT}Data/LPSE/Noisy_rbm/"
params_str = get_params_string(args.model_params)

g = nk.graph.Hypercube(length=args.L, n_dim=args.n_dim, pbc=args.PBC)
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes, total_sz=args.total_sz)
if args.model != "random":
    system = System(args.model, args.model_params, hi, g)
else:
    system = System("TFIM", {"J": 1, "h": 0.5}, hi, g)

seeds = random.sample(range(0, args.n_reps * 100), args.n_reps)

M_z_op = total_magnetization("z", hi)
M_x_op = total_magnetization("x", hi)
H = system.Hamiltonian

re = []
re_M_z = []
re_M_x = []

for seed in tqdm.tqdm(seeds):  
    model_ref = RBM(param_dtype=jnp.float64, alpha=args.alpha)
    sampler_ref = nk.sampler.MetropolisSampler(hi, rule=system.sampler_rule, n_chains=args.n_chains) 
    vs_ref = nk.vqs.MCState(sampler=sampler_ref, model=model_ref, n_samples=args.n_samples, seed=seed, sampler_seed=seed, chunk_size=args.chunk_size)
    
    model = NoisyRBM(sigma_noise=args.sigma, param_dtype=jnp.float64, alpha=args.alpha)
    sampler = nk.sampler.MetropolisSampler(hi, rule=system.sampler_rule, n_chains=args.n_chains) 
    vs = nk.vqs.MCState(sampler=sampler, model=model, n_samples=args.n_samples, seed=seed, sampler_seed=seed, chunk_size=args.chunk_size)

    if args.model != "random":    
        par_folder = f"{PROJECT_ROOT}Data/Parameters/MC/"
        ph = VarHandler(par_folder, args.model, model_ref.label, args.L, args.PBC, args.n_dim, **args.model_params)
        vs_ref.variables = ph.load_variables(vs_ref.variables)
        vs.variables = ph.load_variables(vs_ref.variables)

    vs_ref.sample(n_samples=args.n_therm)
    samples_ref = vs_ref.sample(n_samples=args.n_samples).reshape((-1, g.n_nodes))
    E_ref = vs_ref.expect(H).mean
    M_z_ref = vs_ref.expect(M_z_op).mean
    M_x_ref = vs_ref.expect(M_x_op).mean

    vs.sample(n_samples=args.n_therm)
    samples_delta = vs.sample(n_samples=args.n_samples).reshape((-1, g.n_nodes))
    E_delta = expect(H, samples_delta, model_ref.apply, vs_ref.variables, args.chunk_size).mean
    M_x_delta = expect(M_x_op, samples_delta, model_ref.apply, vs_ref.variables, args.chunk_size).mean
    M_z_delta = expect(M_z_op, samples_delta, model_ref.apply, vs_ref.variables, args.chunk_size).mean

    re.append(compute_RE(E_ref, E_delta))
    re_M_x.append(compute_RE(M_x_ref, M_x_delta))
    re_M_z.append(compute_RE(M_z_ref, M_z_delta))

re = jnp.array(re)
re_M_x = jnp.array(re_M_x)
re_M_z = jnp.array(re_M_z)

re_mean = jnp.mean(re)
re_error = jnp.std(re)
re_M_x_mean = jnp.mean(re_M_x)
re_M_x_error = jnp.std(re_M_x)
re_M_z_mean = jnp.mean(re_M_z)
re_M_z_error = jnp.std(re_M_z)

d_results = {
    "sigma": args.sigma,
    "re_mean": re_mean,
    "re_error": re_error,
    "re_M_x_mean": re_M_x_mean,
    "re_M_x_error": re_M_x_error,
    "re_M_z_mean": re_M_z_mean,
    "re_M_z_error": re_M_z_error,
}
df = pd.DataFrame([d_results])
os.makedirs(out_folder, exist_ok=True)
out = f"{out_folder}/{args.model}{params_str}_L{args.L}_{args.n_dim}dim.csv"
df.to_csv(out, index=False, header=not os.path.isfile(out), mode="a")