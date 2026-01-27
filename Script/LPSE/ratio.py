import netket as nk
import jax.numpy as jnp
import tqdm
import argparse
import ast 
import pandas as pd
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

from mixed_rbm import RBM
from functions import get_params_string
from file_handler import VarHandler
from Custom_nk.Ratio.sampler import rSampler
from Custom_nk.Ratio.MCS import rMCState
from hamiltonian_class import System
from project_paths import PROJECT_ROOT
from parse_utils import str2bool, dict_or_none, none_or_float, print_parsed, none_or_int

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
parser.add_argument("--alpha", default=1, type=int)
parser.add_argument("--chunk_size", default=None, type=none_or_int)
args = parser.parse_args()

print_parsed(args)

out_folder = f"{PROJECT_ROOT}Data/LPSE/Ratio/"
params_str = get_params_string(args.model_params)

dtype_ref = jnp.float64

seed = 123

g = nk.graph.Hypercube(length=args.L, n_dim=args.n_dim, pbc=args.PBC)
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes, total_sz=args.total_sz)
if args.model != "random":
    system = System(args.model, args.model_params, hi, g)
else:
    system = System("TFIM", {"J": 1, "h": 0.5}, hi, g)

model_ref = RBM(alpha=args.alpha, param_dtype=dtype_ref)
sampler_ref = rSampler(hi, rule=system.sampler_rule, n_chains=args.n_chains, dtype=dtype_ref) 
vs_ref = rMCState(sampler=sampler_ref, model=model_ref, n_samples=args.n_samples, seed=seed, sampler_seed=seed, chunk_size=args.chunk_size)

if args.model != "random":    
    par_folder = f"{PROJECT_ROOT}Data/Parameters/MC/"
    ph = VarHandler(par_folder, args.model, model_ref.label, args.L, args.PBC, args.n_dim, **args.model_params)
    vs_ref.variables = ph.load_variables(vs_ref.variables)

vs_ref.sample(n_samples=args.n_therm)
samples, ratios = vs_ref.sample(n_samples=args.n_samples, return_ratios=True)

data = {"Ratio": ratios.reshape((-1))}
df = pd.DataFrame(data)

os.makedirs(out_folder, exist_ok=True)
out_file = f"{out_folder}{args.model}{params_str}_{model_ref.label}_L{args.L}_{args.n_dim}dim.csv"
df.to_csv(out_file, index=False)