import netket as nk
import argparse
import jax

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
from functions import RE
from file_handler import VarHandler
from file_handler import HistoryHandler
from hamiltonian_class import System
from project_paths import PROJECT_ROOT
from parse_utils import str2bool, dict_or_none, none_or_float, none_or_int, print_parsed
from jax_dtype_utils import jax_dtype
from Custom_nk.LPD import LPE_VMC
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
parser.add_argument("--lp_dtype", default=None, type=jax_dtype)
parser.add_argument("--which", default="both", type=str)
parser.add_argument("--L", default=10, type = int)
parser.add_argument("--n_dim", default=1, type = int)
parser.add_argument("--PBC", default=True, type=str2bool)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--n_samples", default=2**12, type=int)
parser.add_argument("--M", default=1000, type = int)
parser.add_argument("--total_sz", default=None, type = none_or_float)
parser.add_argument("--save_parameters", default=True, type = str2bool)
parser.add_argument("--n_chains", default=2**9, type=int)
parser.add_argument("--chunk_size", default=None, type=none_or_int)
parser.add_argument("--sweep_size", default=1, type=int)
parser.add_argument("--n_discard_per_chain", default=5, type=int)
parser.add_argument("--pre_trained", default=False, type=str2bool)
parser.add_argument("--save_history", default=False, type=str2bool)
parser.add_argument("--timeit", default=False, type=str2bool)
parser.add_argument("--use_cg", default=True, type=str2bool)
args = parser.parse_args()

print_parsed(args)

if args.use_cg:
    solver = jax.scipy.sparse.linalg.cg
    solver_name = None
else:
    solver = nk.optimizer.solver.pinv_smooth
    solver_name = "pinv"

if args.lp_dtype.label is not None:
    _folder = f"{args.which.capitalize()}/{args.lp_dtype.label}/"
else:
    _folder = ""

g = nk.graph.Hypercube(length=args.L, n_dim=args.n_dim, pbc=args.PBC)
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes, total_sz=args.total_sz)

model = get_arch(args.arch, g, **args.arch_params)

s = System(args.model, args.model_params, hi, g)

sampler = nk.sampler.MetropolisSampler(hi, rule=s.sampler_rule, n_chains=args.n_chains, sweep_size=args.sweep_size * hi.size) 
opt = nk.optimizer.Sgd(learning_rate = args.lr)

vs = nk.vqs.MCState(sampler, model, n_samples = args.n_samples, chunk_size = args.chunk_size, n_discard_per_chain=args.n_discard_per_chain)

par_folder = f"{PROJECT_ROOT}Data/Parameters{f'_{solver_name}' if solver_name else ''}/{_folder}MC/"
ph = VarHandler(par_folder, s.name, model.label, args.L, args.PBC, args.n_dim, **s.H_par, n_samples=args.n_samples, n_chains=args.n_chains)

if args.pre_trained:
    vs.variables = ph.load_variables(vs.variables)

gs = LPE_VMC(
    hamiltonian=s.Hamiltonian,
    optimizer=opt,
    preconditioner=nk.optimizer.SR(diag_shift=1e-3, holomorphic=False, solver=solver),
    variational_state=vs,
    lp_dtype=args.lp_dtype.dtype,
    which=args.which
)

log = nk.logging.RuntimeLog()
l, t = gs.run(n_iter=args.M, out=log, timeit=args.timeit)

if g.n_nodes < 21:
    E_gs = s.ground_state
    print("Final relative error: ", RE(gs.energy.mean, E_gs))

if args.save_parameters:
    ph.save_variables(vs.variables)

if args.save_history:
    hist_folder = f"{PROJECT_ROOT}Data/History{f'_{solver_name}' if solver_name else ''}/LPE/{_folder}MC/"
    hh = HistoryHandler(hist_folder, s.name, model.label, args.L, args.PBC, args.n_dim, **s.H_par, n_samples=args.n_samples, n_chains=args.n_chains)
    hh.save_history(log, timeit=args.timeit, t = t)