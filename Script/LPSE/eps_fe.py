import netket as nk
import jax.numpy as jnp
import tqdm
import argparse
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
from functions import change_par_dtype, get_params_string
from file_handler import VarHandler
from hamiltonian_class import System
from project_paths import PROJECT_ROOT
from parse_utils import str2bool, dict_or_none, none_or_float, print_parsed

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
parser.add_argument("--total_sz", default=None, type = none_or_float)
parser.add_argument("--alpha", default=1, type=int)
args = parser.parse_args()

print_parsed(args)

out_folder = f"{PROJECT_ROOT}Data/LPSE/"
params_str = get_params_string(args.model_params)

dtype_ref = jnp.float64
dtypes = [jnp.float32, jnp.float16, jnp.bfloat16]
dtypes_labels = ["f32", "f16", "bf16"]

seed = 123

g = nk.graph.Hypercube(length=args.L, n_dim=args.n_dim, pbc=args.PBC)
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes, total_sz=args.total_sz)

model_ref = RBM(alpha=args.alpha, param_dtype=dtype_ref)
vs_ref = nk.vqs.FullSumState(hi, model_ref, seed=seed)

if args.model != "random":    
    par_folder = f"{PROJECT_ROOT}Data/Parameters/MC/"
    ph = VarHandler(par_folder, args.model, f"RBM_alpha{args.alpha}", args.L, args.PBC, args.n_dim, **args.model_params)
    var = ph.load_variables(vs_ref.variables)
    s = System(args.model, args.model_params, hi, g)
else: 
    var = vs_ref.variables
    s = System("TFIM", {"J": 1, "h": 0.5}, hi, g)

H = s.Hamiltonian

sigmas = hi.all_states()
log_pdf_ref = 2 * model_ref.apply(var, sigmas).real

eps = []
dict = {}

print("Computing epsilon for each dtype...")
for i, dtype in tqdm.tqdm(enumerate(dtypes)): 
    model = RBM(param_dtype=dtype, alpha=args.alpha)
    log_pdf_delta = 2 * model.apply({"params": change_par_dtype(var["params"], dtype)}, sigmas).real

    eps.append(log_pdf_delta - log_pdf_ref)
    dict[dtypes_labels[i]] = eps[i]

eps = jnp.array(eps).reshape((len(dtypes), -1))
df = pd.DataFrame(dict)
df.to_csv(f"{out_folder}Eps/FE/{args.model}{params_str}_{model.label}_L{args.L}_{args.n_dim}dim.csv", index=False)