import netket as nk
import jax.numpy as jnp
import tqdm
import argparse
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

from Archs.noisy_rbm import NoisyRBM
from Archs.rbm import RBM
from functions import get_params_string, compute_RE
from file_handler import VarHandler
from Custom_nk.expect import expect
from hamiltonian_class import System
from project_paths import PROJECT_ROOT
from parse_utils import str2bool, print_parsed, none_or_int
from save_utils import create_or_append

parser = argparse.ArgumentParser()
parser.add_argument("--h", default=0.5, type=float)
parser.add_argument("--L", default=12, type=int)
parser.add_argument("--n_dim", default=1, type=int)
parser.add_argument("--PBC", default=True, type=str2bool)
parser.add_argument("--n_samples", default=2**17, type=int)
parser.add_argument("--n_therm", default=2**12, type = int)
parser.add_argument("--n_chains", default=16, type=int)
parser.add_argument("--n_reps", default=5, type=int)
parser.add_argument("--alpha", default=1, type=int)
parser.add_argument("--chunk_size", default=None, type=none_or_int)
parser.add_argument("--sigma", default=1e-3, type=float)
args = parser.parse_args()

print_parsed(args)

out_folder = f"{PROJECT_ROOT}Data/LPSE/Noisy_rbm/Dispersion/"
par_folder = f"{PROJECT_ROOT}Data/Parameters/MC/"

model_name = "TFIM"
model_params = {"J":1, "h": args.h}
params_str = get_params_string(model_params)

g = nk.graph.Hypercube(length=args.L, n_dim=args.n_dim, pbc=args.PBC)
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
system = System("TFIM", {"J": 1, "h": args.h}, hi, g)

seeds = random.sample(range(0, args.n_reps * 100), args.n_reps)

H = system.Hamiltonian
H_heis = System("Heisenberg", {"J": 1}, hi, g).Hamiltonian

re_e = []
re_heis = []

for seed in tqdm.tqdm(seeds):  
    model_ref = RBM(param_dtype=jnp.float64, alpha=args.alpha)
    sampler_ref = nk.sampler.MetropolisSampler(hi, rule=system.sampler_rule, n_chains=args.n_chains) 
    vs_ref = nk.vqs.MCState(sampler=sampler_ref, model=model_ref, n_samples=args.n_samples, seed=seed, sampler_seed=seed, chunk_size=args.chunk_size)
    
    model = NoisyRBM(sigma_noise=args.sigma, param_dtype=jnp.float64, alpha=args.alpha)
    sampler = nk.sampler.MetropolisSampler(hi, rule=system.sampler_rule, n_chains=args.n_chains) 
    vs = nk.vqs.MCState(sampler=sampler, model=model, n_samples=args.n_samples, seed=seed, sampler_seed=seed, chunk_size=args.chunk_size)
 
    ph = VarHandler(par_folder, model_name, model_ref.label, args.L, args.PBC, args.n_dim, **model_params)
    vs_ref.variables = ph.load_variables(vs_ref.variables)
    vs.variables = ph.load_variables(vs_ref.variables)

    vs_ref.sample(n_samples=args.n_therm)
    samples_ref = vs_ref.sample(n_samples=args.n_samples).reshape((-1, g.n_nodes))
    E_ref = vs_ref.expect(H).mean
    E_heis_ref = vs_ref.expect(H_heis).mean

    vs.sample(n_samples=args.n_therm)
    samples_delta = vs.sample(n_samples=args.n_samples).reshape((-1, g.n_nodes))
    E_delta = expect(H, samples_delta, model_ref.apply, vs_ref.variables, args.chunk_size).mean
    E_heis_delta = expect(H_heis, samples_delta, model_ref.apply, vs_ref.variables, args.chunk_size).mean

    re_e.append(compute_RE(E_ref, E_delta))
    re_heis.append(compute_RE(E_heis_ref, E_heis_delta))

re_e = jnp.array(re_e)
re_heis = jnp.array(re_heis)

data = {
    "sigma": args.sigma,
    "re_e_mean": jnp.mean(re_e),
    "re_e_error": jnp.std(re_e),
    "re_heis_mean": jnp.mean(re_heis),
    "re_heis_error": jnp.std(re_heis)
}

create_or_append(data, out_folder, f"{model_name}{params_str}_L{args.L}_{args.n_dim}dim.csv")