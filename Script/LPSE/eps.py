import netket as nk
import jax.numpy as jnp
import tqdm
import argparse
import ast 
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
from Custom_nk.expect import get_local_sum, expect
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
parser.add_argument("--n_samples", default=2**17)
parser.add_argument("--total_sz", default=None, type = none_or_float)
parser.add_argument("--n_therm", default=2**12, type = float)
parser.add_argument("--n_chains", default=16, type=int)
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

PDF_ref = jnp.exp(log_pdf_ref)
PDF_ref = PDF_ref / jnp.sum(PDF_ref)
exp_eps_delta = jnp.exp(eps)
norm_delta = jnp.einsum("j, lj -> l", PDF_ref, exp_eps_delta)
bias = 1 - jnp.einsum("jk, j -> jk", exp_eps_delta, 1 / norm_delta)
local_sums = get_local_sum(H.to_jax_operator(), sigmas, model_ref.apply, var)

D_delta_th = jnp.einsum("j, j, kj -> k", PDF_ref, local_sums, bias)

mean_delta = []
error_delta = []

sampler_ref = nk.sampler.MetropolisSampler(hi, rule=s.sampler_rule, n_chains=args.n_chains, dtype=dtype_ref) 
vs_ref = nk.vqs.MCState(sampler=sampler_ref, model=model_ref, n_samples=args.n_samples, seed=seed, sampler_seed=seed)    

vs_ref.parameters = var["params"]
vs_ref.sample(n_samples = args.n_therm)
vs_ref.sample()
E_ref = vs_ref.expect(H)

mean_ref = E_ref.mean
error_ref = E_ref.error_of_mean

print("Computing the energy...")
for dtype in tqdm.tqdm(dtypes):
    model = RBM(param_dtype = dtype, alpha=args.alpha)
    sampler = nk.sampler.MetropolisSampler(hi, rule=s.sampler_rule, n_chains=args.n_chains, dtype=dtype) 
    vs_delta = nk.vqs.MCState(sampler=sampler, model=model, n_samples=args.n_samples, seed=seed, sampler_seed=seed)
    vs_delta.parameters = change_par_dtype(var["params"], dtype)
    vs_delta.sample(n_samples = args.n_therm)
    samples_delta = vs_delta.sample().reshape((-1, g.n_nodes))
    E_delta = expect(H, samples_delta, model_ref.apply, vs_ref.variables)

    mean_delta.append(E_delta.mean)
    error_delta.append(E_delta.error_of_mean)

mean_delta = jnp.array(mean_delta)
error_delta = jnp.array(error_delta)

D_delta = mean_ref - mean_delta
sigma_D_delta = (jnp.sqrt(error_ref ** 2 + error_delta ** 2))
z = jnp.abs((D_delta_th - D_delta) / (sigma_D_delta + 1e-10))

d_results = {
    "dtypes": dtypes_labels,
    "E_mean_ref": mean_ref,
    "E_error_ref": error_ref,
    "E_mean_delta": mean_delta,
    "E_error_delta": error_delta, 
    "z": z
}
df = pd.DataFrame(d_results)
df.to_csv(f"{out_folder}Energy/FE/{args.model}{params_str}_{model.label}_L{args.L}_{args.n_dim}dim.csv", index=False)