import netket as nk
import argparse
import tqdm
import numpy as np
import os
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

from Archs.noisy_rbm import NoisyRBM
from functions import get_params_string
from file_handler import VarHandler
from hamiltonian_class import System
from project_paths import PROJECT_ROOT
from parse_utils import str2bool, none_or_int, print_parsed
from Custom_nk.expect import expect
from Custom_nk.LPS import Custom_sampler
from save_utils import create_or_append

parser = argparse.ArgumentParser()
parser.add_argument("--L", default=10, type=int)
parser.add_argument("--n_dim", default=1, type=int)
parser.add_argument("--PBC", default=True, type=str2bool)
parser.add_argument("--n_chains", default=16, type=int)
parser.add_argument("--n_therm", default=2**17, type=int)
parser.add_argument("--n_samples", default=2**12, type=int)
parser.add_argument("--chunk_size", default=None, type=none_or_int)
parser.add_argument("--alpha", default=1, type=int)
parser.add_argument("--h", default=0.5, type=float)
args = parser.parse_args()

print_parsed(args)

seed = 123

sigmas = np.concatenate(([0], np.logspace(-3, 3, 100)))

par_folder = f"{PROJECT_ROOT}Data/Parameters/MC/"
g = nk.graph.Hypercube(length=args.L, n_dim=args.n_dim, pbc=args.PBC)
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

system = System("TFIM", {"J": 1, "h": args.h}, hi, g)

H = system.Hamiltonian
H_heis = System("Heisenberg", {"J": 1}, hi, g).Hamiltonian

model_ref = NoisyRBM(sigma_noise=0)
sampler_ref = Custom_sampler(hi, rule=system.sampler_rule, n_chains=args.n_chains)
vs_ref = nk.vqs.MCState(sampler=sampler_ref, model=model_ref, n_samples=args.n_samples, seed=seed, sampler_seed=seed, chunk_size=args.chunk_size)
ph = VarHandler(par_folder, system.name, f"RBM_alpha{args.alpha}", args.L, args.PBC, args.n_dim, **system.H_par)
vs_ref.variables = ph.load_variables(vs_ref.variables)

results = []

for s in tqdm.tqdm(range(len(sigmas))):
    model = NoisyRBM(sigma_noise=sigmas[s])

    sampler = Custom_sampler(hi, rule=system.sampler_rule, n_chains=args.n_chains) 
    vs = nk.vqs.MCState(sampler=sampler, model=model, n_samples=args.n_samples, seed=seed, sampler_seed=seed, chunk_size=args.chunk_size)
    vs.variables = vs_ref.variables
    vs.sample(n_samples=args.n_therm)
    samples = vs.sample(n_samples=args.n_samples)

    results.append({
        "Sigma": sigmas[s],
        "Acceptance": vs.sampler_state.acceptance.item()
    })
params_str = get_params_string({"J": 1, "h": args.h})
out_folder = f"{PROJECT_ROOT}Data/LPSE/Noisy_rbm/Acceptance/"
out = out_folder + f"TFIM{params_str}_L{args.L}_{args.n_dim}dim.csv"
os.makedirs(out_folder, exist_ok=True)
df = pd.DataFrame(results)
df.to_csv(out, index=False)