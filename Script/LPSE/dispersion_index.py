import netket as nk
import argparse
import tqdm
import numpy as np
import jax.numpy as jnp
import pandas as pd

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
from functions import RE, get_params_string, change_par_dtype
from file_handler import VarHandler
from hamiltonian_class import System
from observables import total_magnetization
from project_paths import PROJECT_ROOT
from parse_utils import str2bool, dict_or_none, none_or_float, none_or_int, print_parsed
from Custom_nk.expect import expect
from Custom_nk.LPS import Custom_sampler

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="TFIM", type=str)
parser.add_argument(
    "--model_params",
    default='{"J": 1, "h": 0.5}',
    type=dict_or_none,
    help='Dictionary as a string, e.g., \'{"J": 1, "h": 0.5}\' or None',
)
parser.add_argument("--L", default=10, type=int)
parser.add_argument("--n_dim", default=1, type=int)
parser.add_argument("--PBC", default=True, type=str2bool)
parser.add_argument("--n_therm", default=2**17, type=int)
parser.add_argument("--n_samples", default=2**17, type=int)
parser.add_argument("--n_chains", default=16, type=int)
parser.add_argument("--total_sz", default=None, type = none_or_float)
parser.add_argument("--chunk_size", default=None, type=none_or_int)
parser.add_argument("--alpha", default=1, type=int)
args = parser.parse_args()

print_parsed(args)

par_folder = f"{PROJECT_ROOT}Data/Parameters/MC/"
g = nk.graph.Hypercube(length=args.L, n_dim=args.n_dim, pbc=args.PBC)
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes, total_sz=args.total_sz)

if args.model != "random":
    system = System(args.model, args.model_params, hi, g)
else:
    system = System("TFIM", {"J": 1, "h": 0.1}, hi, g)

model = RBM(alpha=args.alpha)
sampler = Custom_sampler(hi, rule=system.sampler_rule, n_chains=args.n_chains) 
vs = nk.vqs.MCState(sampler=sampler, model=model, n_samples=args.n_samples, chunk_size=args.chunk_size)

if args.model != "random":
    ph = VarHandler(par_folder, system.name, model.label, args.L, args.PBC, args.n_dim, **system.H_par)
    vs.variables = ph.load_variables(vs.variables)

vs.sample(n_samples=args.n_therm)
samples = vs.sample(n_samples=args.n_samples).reshape((-1, g.n_nodes))
dispersion_index = len(jnp.unique(samples, axis=0)) / min(2 ** g.n_nodes, args.n_samples)

if args.model != "random":
    results = args.model_params
    results["dispersion_index"] = dispersion_index

else: 
    results = {"dispersion_index": dispersion_index}

df = pd.DataFrame([results])

params_str = get_params_string(args.model_params)
out_folder = f"{PROJECT_ROOT}Data/LPSE/Dispersion_index/"
Path(out_folder).mkdir(parents=True, exist_ok=True)
df.to_csv(f"{out_folder}{args.model}_{model.label}_L{args.L}_{args.n_dim}dim.csv", index=False)