import sys
from pathlib import Path
import jax
import jax.numpy as jnp

current_path = Path(__file__).resolve()
project_root = None
for parent in current_path.parents:
    if (parent / ".project_root").exists():
        project_root = parent
        break
if project_root is None:
    raise RuntimeError("Could not find the project root. Make sure a '.project_root' file exists.")
sys.path.append(str(project_root))

from Archs.rbm import RBM, RBMSymm
from Archs.res_cnn import ResCNN

def get_arch(name, g, help=False, arch_par=None, **kwargs):
    if arch_par is None:
        return _get_arch(name, g, help, **kwargs)
    else:
        return _get_arch(name, g, help, **arch_par)
    
def _get_arch(name, g, help, **arch_par):
    mapping = {
        "RBM": lambda: RBM(**arch_par),
        "RBMSymm": lambda: RBMSymm(symmetries=g.space_group(), **arch_par),
        "ResCNN": lambda: ResCNN(linear_size=g.extent[0], **arch_par)
    }

    header = "Hyper-parameters of the model: \n"
    info = {
        "RBM": header + 
            "\talpha: density of parameters",
        "RBMSymm": header + 
            "\talpha: density of parameters",
        "ResCNN": header +
            "\tn_res_blocks: number of residual blocks\n"
            "\tfilters: number of filters\n"
            "\tkernel_shape: shape of the kernel (n, n)\n"
            "\tupcast_sums: if True all the sums are computed in at least float32"
    }

    for key, value in mapping.items():
        if key.lower() == name.lower():
            if help: 
                print(info[name])
            return value()
    raise NotImplementedError(f"Take an available architecture between: {', '.join(mapping.keys())}, got: {name.lower()}")

def count_parameters(params):
    """Counts the total number of parameters in a Flax Linen model."""
    def count_leaf(x):
        return x.size

    return sum(jax.tree_util.tree_leaves(jax.tree_map(count_leaf, params)))

def change_par_dtype(par, dtype):
    def _change(x):
        if isinstance(x, jnp.ndarray):
            return x.astype(dtype)
        return x
    
    return jax.tree_util.tree_map(_change, par)