import netket as nk
import jax.numpy as jnp
import tqdm
import numpy as np
import pandas as pd
import jax.tree_util as tree_util
import argparse
import ast

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
from mixed_rbm import RBM
from functions import change_par_dtype, spin_to_number, MC_KL, FE_KL, get_PDF, model_to_PDF, get_params_string
from file_handler import VarHandler
from hamiltonian_class import System
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
parser.add_argument("--exp_n_samples_max", default=20, type=int)
parser.add_argument("--total_sz", default=None, type = none_or_float)
parser.add_argument("--n_therm", default=2**12, type = int)
parser.add_argument("--FE", default=False, type=str2bool)
parser.add_argument("--alpha", default=1, type=int)
args = parser.parse_args()

print_parsed(args)

out_folder = f"{PROJECT_ROOT}Data/LPSE/"
params_str = get_params_string(args.model_params)

dtype_ref = jnp.float64
dtypes = [jnp.float64, jnp.float32, jnp.float16, jnp.bfloat16]
dtype_labels = ["f64", "f32", "f16", "bf16"]

n_samples = 2 ** np.arange(10, args.exp_n_samples_max + 1)
n_chains = n_samples // (2 ** 5)
seed = 123

g = nk.graph.Hypercube(length=args.L, n_dim=args.n_dim, pbc=args.PBC)
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes, total_sz=args.total_sz)
if args.model != "random":
    system = System(args.model, args.model_params, hi, g)
else:
    system = System("TFIM", {"J": 1, "h": 0.5}, hi, g)

model_ref = RBM(alpha=args.alpha, param_dtype = dtype_ref)
sampler_ref = nk.sampler.MetropolisSampler(hi, rule=system.sampler_rule, n_chains=n_chains[-1], dtype=dtype_ref) 
vs_ref = nk.vqs.MCState(sampler=sampler_ref, model=model_ref, n_samples=n_samples[-1], seed=seed, sampler_seed=seed)

if args.model != "random":    
    par_folder = f"{PROJECT_ROOT}Data/Parameters/MC/"
    ph = VarHandler(par_folder, args.model, model_ref.label, args.L, args.PBC, args.n_dim, **args.model_params)
    par = ph.load_parameters(vs_ref.variables)
else: 
    par = vs_ref.parameters

vs_ref.parameters = tree_util.tree_map(lambda x: x.copy(), par)

if args.FE:
    samples_ref = hi.all_states()
    log_PDF_ref, PDF_ref, norm_ref = model_to_PDF(model_ref.apply, samples_ref[::-1], vs_ref.variables)

    full_size = int(2 ** g.n_nodes)
    if hi.n_states != full_size:
        indices = spin_to_number((samples_ref + 1) // 2)
        log_PDF_pad = jnp.zeros((full_size))
        PDF_pad = jnp.zeros((full_size))
        PDF_ref = PDF_pad.at[indices].set(PDF_ref)
        log_PDF_ref = log_PDF_pad.at[indices].set(log_PDF_ref)

    get_KL = lambda x: FE_KL(PDF_ref, log_PDF_ref, norm_ref, x)
    out_folder += "KL_p_fe/"

else:
    vs_ref.sample(n_samples=args.n_therm)
    samples_ref = vs_ref.sample().reshape((-1, g.n_nodes))

    get_KL = lambda x: MC_KL(model_ref.apply, vs_ref.variables, x, samples_ref)
    out_folder += "KL_p_mc/"

KLs = []

print("Computing the KL-divergence for each dtype...")
for dtype in tqdm.tqdm(dtypes):
    model = RBM(alpha=args.alpha, param_dtype = dtype)
    
    for i, N in enumerate(n_samples):
        sampler = nk.sampler.MetropolisSampler(hi, rule=system.sampler_rule, n_chains=n_chains[i], dtype=dtype) 
        vs = nk.vqs.MCState(sampler=sampler, model=model, n_samples=n_samples[i], seed=seed, sampler_seed=seed)
        vs.parameters = change_par_dtype(par, dtype)

        vs.sample(n_samples = args.n_therm)
        samples = vs.sample().reshape((-1, args.L))

        s = ((samples + 1) // 2).astype(jnp.int64)
        int_samples = spin_to_number(s)
        PDF = get_PDF(int_samples, args.L)
        KLs.append(get_KL(PDF))
        
KLs = np.array(KLs).reshape((len(dtypes), -1))

data = []
for dtype_idx, dtype_name in enumerate(dtype_labels):
    for sample_idx, N in enumerate(n_samples):
        data.append({
            "dtype": dtype_name,
            "n_samples": N,
            "KL": KLs[dtype_idx, sample_idx]
        })

df = pd.DataFrame(data)
df.to_csv(f"{out_folder}{args.model}{params_str}_{model.label}_L{args.L}_{args.n_dim}dim.csv", index=False)