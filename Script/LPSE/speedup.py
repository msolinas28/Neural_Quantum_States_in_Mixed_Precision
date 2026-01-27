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

from functions import change_par_dtype, get_params_string, time_sampling
from file_handler import VarHandler
from hamiltonian_class import System
from project_paths import PROJECT_ROOT
from parse_utils import str2bool, dict_or_none, none_or_float, print_parsed, none_or_int
from Custom_nk.LPS import Custom_sampler
from Archs.arch_utils import get_arch

parser = argparse.ArgumentParser()
parser.add_argument("--arch", default="RBM", type=str)
parser.add_argument(
    "--arch_params",
    default='{"alpha": 1}',
    type=dict_or_none,
    help='Dictionary as a string, e.g., \'{"alpha": 1}\' or None',
)
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
parser.add_argument("--n_reps", default=10, type=int)
parser.add_argument("--chunk_size", default=None, type=none_or_int)
args = parser.parse_args()

print_parsed(args)

out_folder = f"{PROJECT_ROOT}Data/LPSE/Speedup/"
params_str = get_params_string(args.model_params)

dtype_ref = jnp.float64
dtypes = [jnp.float32, jnp.float16, jnp.bfloat16]
dtypes_labels = ["f32", "f16", "bf16"]

seed = 123

g = nk.graph.Hypercube(length=args.L, n_dim=args.n_dim, pbc=args.PBC)
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes, total_sz=args.total_sz)
if args.model != "random":
    system = System(args.model, args.model_params, hi, g)
else:
    system = System("TFIM", {"J": 1, "h": 0.5}, hi, g)

model_ref = get_arch(args.arch, g, param_dtype=dtype_ref, **args.arch_params)
sampler_ref = Custom_sampler(hi, rule=system.sampler_rule, n_chains=args.n_chains, dtype_ratio=dtype_ref) 
vs_ref = nk.vqs.MCState(sampler=sampler_ref, model=model_ref, n_samples=args.n_samples, seed=seed, sampler_seed=seed, chunk_size=args.chunk_size)

if args.model != "random":    
    par_folder = f"{PROJECT_ROOT}Data/Parameters/MC/"
    ph = VarHandler(par_folder, args.model, model_ref.label, args.L, args.PBC, args.n_dim, **args.model_params)
    vs_ref.variables = ph.load_variables(vs_ref.variables)

vs_ref.sample(n_samples = args.n_therm)
time_ref_mean, time_ref_err = time_sampling(vs_ref.sample, args.n_samples, args.n_reps)

time_delta_mean = []
time_delta_err = []
speedup_mean = []
speedup_err = []

print("Timing each dtype...")
for i, dtype in tqdm.tqdm(enumerate(dtypes)):  
    model = get_arch(args.arch, g, param_dtype=dtype, **args.arch_params)
    sampler = Custom_sampler(hi, rule=system.sampler_rule, n_chains=args.n_chains, dtype_ratio=dtype)
    vs = nk.vqs.MCState(sampler=sampler, model=model, n_samples=args.n_samples, seed=seed, sampler_seed=seed, chunk_size=args.chunk_size)
    vs.parameters = change_par_dtype(vs_ref.parameters, dtype)
    vs.sample(n_samples=args.n_therm)
    t_mean, t_err = time_sampling(vs.sample, args.n_samples, args.n_reps)

    time_delta_mean.append(t_mean)
    time_delta_err.append(t_err)
    speedup_mean.append(time_ref_mean / t_mean)
    speedup_err.append(jnp.sqrt((t_err/t_mean)**2 + (time_ref_err/time_ref_mean)**2))

time_delta_mean = jnp.array(time_delta_mean)
time_delta_err = jnp.array(time_delta_err)
speedup_mean = jnp.array(speedup_mean)
speedup_err = jnp.array(speedup_err)

d_results = {
    "dtypes": dtypes_labels,
    "time_ref_mean": time_ref_mean,
    "time_ref_err": time_ref_err,
    "time_delta_mean": time_delta_mean,
    "time_delta_err": time_delta_err,
    "speedup_mean": speedup_mean,
    "speedup_err": speedup_err
}
df = pd.DataFrame(d_results)
df.to_csv(f"{out_folder}{args.model}{params_str}_{model.label}_L{args.L}_{args.n_dim}dim.csv", index=False)