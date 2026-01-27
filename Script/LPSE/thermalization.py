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
parser.add_argument("--L", default=10, type = int)
parser.add_argument("--n_dim", default=1, type = int)
parser.add_argument("--PBC", default=True, type=str2bool)
parser.add_argument("--n_chains", default=16, type=int)
parser.add_argument("--total_sz", default=None, type = none_or_float)
parser.add_argument("--chunk_size", default=None, type=none_or_int)
parser.add_argument("--alpha", default=1, type=int)
args = parser.parse_args()

print_parsed(args)

seed = 123
dtype_ref = jnp.float64
dtypes = [jnp.float64, jnp.float32, jnp.float16, jnp.bfloat16]
dtypes_label = ["f64", "f32", "f16", "bf16"]

N_therm = 2 ** np.arange(10, 17)
if args.n_chains > N_therm[0]:
    raise ValueError(f"The number of chains has to be smaller or equal than {N_therm[0]}!")

par_folder = f"{PROJECT_ROOT}Data/Parameters/MC/"
g = nk.graph.Hypercube(length=args.L, n_dim=args.n_dim, pbc=args.PBC)
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes, total_sz=args.total_sz)

if args.model != "random":
    system = System(args.model, args.model_params, hi, g)
else:
    system = System("TFIM", {"J": 1, "h": 0.5}, hi, g)

H = system.Hamiltonian
M_z = total_magnetization("z", hi)

model_ref = RBM(alpha=args.alpha, param_dtype=dtype_ref)
sampler_ref = Custom_sampler(hi, rule=system.sampler_rule, n_chains=args.n_chains, dtype_ratio=dtype_ref) 
vs_ref = nk.vqs.MCState(sampler=sampler_ref, model=model_ref, n_samples=N_therm[0], seed=seed, sampler_seed=seed, chunk_size=args.chunk_size)

if args.model != "random":
    ph = VarHandler(par_folder, system.name, model_ref.label, args.L, args.PBC, args.n_dim, **system.H_par)
    vs_ref.variables = ph.load_variables(vs_ref.variables)

results = []

for d in tqdm.tqdm(range(len(dtypes))):
    model = RBM(alpha=args.alpha, param_dtype=dtypes[d])
    par = change_par_dtype(vs_ref.parameters, dtypes[d])

    for n_therm in N_therm:
        sampler = Custom_sampler(hi, rule=system.sampler_rule, n_chains=args.n_chains, dtype_ratio=dtypes[d]) 
        vs = nk.vqs.MCState(sampler=sampler, model=model, n_samples=n_therm, seed=seed, sampler_seed=seed, chunk_size=args.chunk_size)
        vs.parameters = par
        samples = vs.sample()

        m_z = expect(M_z, samples, model_ref.apply, vs_ref.variables, args.chunk_size)
        e = expect(H, samples, model_ref.apply, vs_ref.variables, args.chunk_size)

        results.append({
            "dtype": dtypes_label[d],
            "n_therm": n_therm,
            "E_mean": e.mean,
            "E_error": e.error_of_mean,
            "M_z_mean": m_z.mean,
            "M_z_error": m_z.error_of_mean
        })

df = pd.DataFrame(results)

params_str = get_params_string(args.model_params)
out_folder = f"{PROJECT_ROOT}Data/LPSE/Thermalization/"
Path(out_folder).mkdir(parents=True, exist_ok=True)
df.to_csv(f"{out_folder}{args.model}{params_str}_{model.label}_L{args.L}_{args.n_dim}dim.csv", index=False)