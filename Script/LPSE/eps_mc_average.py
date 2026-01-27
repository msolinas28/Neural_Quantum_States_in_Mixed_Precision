import netket as nk
import jax.numpy as jnp
import tqdm
import argparse
import ast 
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

from mixed_rbm import RBM
from functions import change_par_dtype, get_params_string, compute_RE
from file_handler import VarHandler
from Custom_nk.expect import get_local_sum, expect
from Custom_nk.LPS import Custom_sampler
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
args = parser.parse_args()

print_parsed(args)

out_folder = f"{PROJECT_ROOT}Data/LPSE/"
params_str = get_params_string(args.model_params)

dtype_ref = jnp.float64
dtypes = [jnp.float32, jnp.float16, jnp.bfloat16]
dtypes_labels = ["f32", "f16", "bf16"]

g = nk.graph.Hypercube(length=args.L, n_dim=args.n_dim, pbc=args.PBC)
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes, total_sz=args.total_sz)
if args.model != "random":
    system = System(args.model, args.model_params, hi, g)
else:
    system = System("TFIM", {"J": 1, "h": 0.5}, hi, g)

model_ref = RBM(alpha=args.alpha, param_dtype=dtype_ref)

seeds = random.sample(range(0, args.n_reps * 100), args.n_reps)

M_z_op = total_magnetization("z", hi)
M_x_op = total_magnetization("x", hi)

sigma = []
re = []
acceptance = []
re_M_z = []
re_M_x = []

for seed in tqdm.tqdm(seeds):  
    sampler_ref = nk.sampler.MetropolisSampler(hi, rule=system.sampler_rule, n_chains=args.n_chains, dtype=dtype_ref) 
    vs_ref = nk.vqs.MCState(sampler=sampler_ref, model=model_ref, n_samples=args.n_samples, seed=seed, sampler_seed=seed, chunk_size=args.chunk_size)

    if args.model != "random":    
        par_folder = f"{PROJECT_ROOT}Data/Parameters/MC/"
        ph = VarHandler(par_folder, args.model, model_ref.label, args.L, args.PBC, args.n_dim, **args.model_params)
        vs_ref.variables = ph.load_variables(vs_ref.variables)

    H = system.Hamiltonian

    vs_ref.sample(n_samples=args.n_therm)
    samples_ref = vs_ref.sample(n_samples=args.n_samples).reshape((-1, g.n_nodes))
    E_ref = vs_ref.expect(H).mean
    acceptance_ref = vs_ref.sampler_state.acceptance
    M_z_ref = vs_ref.expect(M_z_op).mean
    M_x_ref = vs_ref.expect(M_x_op).mean

    print("Computing epsilon for each dtype...")
    for i, dtype in enumerate(dtypes):  
        model = RBM(param_dtype=dtype, alpha=args.alpha)
        sampler = Custom_sampler(hi, rule=system.sampler_rule, n_chains=args.n_chains, dtype_ratio=dtype)
        vs = nk.vqs.MCState(sampler=sampler, model=model, n_samples=args.n_samples, seed=seed, sampler_seed=seed, chunk_size=args.chunk_size)
        vs.parameters = change_par_dtype(vs_ref.parameters, dtype)
        vs.sample(n_samples=args.n_therm)
        samples_delta = vs.sample(n_samples=args.n_samples).reshape((-1, g.n_nodes))

        log_pdf_ref = 2 * model_ref.apply(vs_ref.variables, samples_ref).real
        log_pdf_delta = 2 * model.apply(vs.variables, samples_ref).real

        eps = log_pdf_delta - log_pdf_ref
        E_delta = expect(H, samples_delta, model_ref.apply, vs_ref.variables, args.chunk_size).mean
        M_x_delta = expect(M_x_op, samples_delta, model_ref.apply, vs_ref.variables, args.chunk_size).mean
        M_z_delta = expect(M_z_op, samples_delta, model_ref.apply, vs_ref.variables, args.chunk_size).mean

        sigma.append(jnp.std(jnp.array(eps)))
        re.append(compute_RE(E_ref, E_delta))
        acceptance.append(vs.sampler_state.acceptance)
        re_M_x.append(compute_RE(M_x_ref, M_x_delta))
        re_M_z.append(compute_RE(M_z_ref, M_z_delta))

re = jnp.array(re).reshape((args.n_reps, len(dtypes)))
sigma = jnp.array(sigma).reshape((args.n_reps, len(dtypes)))
acceptance = jnp.array(acceptance).reshape((args.n_reps, len(dtypes)))
re_M_x = jnp.array(re_M_x).reshape((args.n_reps, len(dtypes)))
re_M_z = jnp.array(re_M_z).reshape((args.n_reps, len(dtypes)))

re_mean = jnp.mean(re, axis=0)
re_error = jnp.std(re, axis=0)
sigma_mean = jnp.mean(sigma, axis=0)
sigma_error = jnp.std(sigma, axis=0)
acceptance_mean = jnp.mean(acceptance, axis=0)
acceptance_error = jnp.std(acceptance, axis=0)
re_M_x_mean = jnp.mean(re_M_x, axis=0)
re_M_x_error = jnp.std(re_M_x, axis=0)
re_M_z_mean = jnp.mean(re_M_z, axis=0)
re_M_z_error = jnp.std(re_M_z, axis=0)

d_results = {
    "dtypes": dtypes_labels,
    "re_mean": re_mean,
    "re_error": re_error,
    "sigma_mean": sigma_mean,
    "sigma_error": sigma_error,
    "acceptance_ref": acceptance_ref,
    "acceptance_mean": acceptance_mean,
    "acceptance_error": acceptance_error, 
    "re_M_x_mean": re_M_x_mean,
    "re_M_x_error": re_M_x_error,
    "re_M_z_mean": re_M_z_mean,
    "re_M_z_error": re_M_z_error,
}
df = pd.DataFrame(d_results)
df.to_csv(f"{out_folder}Average/{args.model}{params_str}_{model.label}_L{args.L}_{args.n_dim}dim.csv", index=False)