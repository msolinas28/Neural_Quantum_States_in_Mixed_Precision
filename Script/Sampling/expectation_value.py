import netket as nk
import argparse
import tqdm
import numpy as np
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
from functions import RE, get_params_string
from file_handler import VarHandler
from hamiltonian_class import System
from observables import total_magnetization
from project_paths import PROJECT_ROOT
from parse_utils import str2bool, dict_or_none, none_or_float, none_or_int, print_parsed

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
parser.add_argument("--n_samples", default=2**17, type=int)
parser.add_argument("--n_therm", default=2**12, type=int)
parser.add_argument("--total_sz", default=None, type = none_or_float)
parser.add_argument("--chunk_size", default=None, type=none_or_int)
parser.add_argument("--alpha", default=1, type=int)
args = parser.parse_args()

print_parsed(args)

N_chains = 2 ** np.arange(4, int(np.log2(args.n_samples)) - 2)
N_discard_per_chain = [5, 10, 100]

par_folder = f"{PROJECT_ROOT}Data/Parameters/MC/"
g = nk.graph.Hypercube(length=args.L, n_dim=args.n_dim, pbc=args.PBC)
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes, total_sz=args.total_sz)

model = RBM(alpha=args.alpha)

s = System(args.model, args.model_params, hi, g)
H = s.Hamiltonian
M_z = total_magnetization("z", hi)

ph = VarHandler(par_folder, s.name, model.label, args.L, args.PBC, args.n_dim, **s.H_par)

results = []

for n_discard_per_chain in tqdm.tqdm(N_discard_per_chain):
    for n_chains in N_chains:
        sampler = nk.sampler.MetropolisSampler(hi, rule=s.sampler_rule, n_chains=n_chains) 
        vs = nk.vqs.MCState(sampler, model, n_samples=args.n_samples, chunk_size=args.chunk_size, n_discard_per_chain=n_discard_per_chain)
        vs.parameters = ph.load_parameters(vs.variables)
        vs.sample(n_samples=args.n_therm)

        m_z = vs.expect(M_z)
        e = vs.expect(H)

        results.append({
            "n_chains": n_chains,
            "n_discard_per_chain": n_discard_per_chain,
            "E_mean": e.mean,
            "E_error": e.error_of_mean,
            "tau_E": e.tau_corr,
            "R_E": e.R_hat,
            "M_z_mean": m_z.mean,
            "M_z_error": m_z.error_of_mean,
            "tau_M_z": m_z.tau_corr,
            "R_M_z": m_z.R_hat
        })

df = pd.DataFrame(results)

params_str = get_params_string(args.model_params)
out_folder = f"{PROJECT_ROOT}Data/Sampling/Expectation_value/"
Path(out_folder).mkdir(parents=True, exist_ok=True)
df.to_csv(f"{out_folder}{args.model}{params_str}_{model.label}_L{args.L}_{args.n_dim}dim.csv", index=False)