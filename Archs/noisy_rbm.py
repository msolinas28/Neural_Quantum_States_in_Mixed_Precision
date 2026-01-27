import jax
import jax.numpy as jnp
from netket.models import RBM
from functools import partial
from flax import linen as nn

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

from functions import spin_to_number

def _gen_noise(integer_hash, base_key = jax.random.PRNGKey(0)):
  key = jax.random.fold_in(base_key, integer_hash)
  return jax.random.normal(key)

@jax.jit
def gen_noise(keys):
  keys = jnp.atleast_1d(keys)
  return jax.vmap(_gen_noise)(keys)

@partial(jax.jit, static_argnames=["sigma"])
def sample_to_noise(sample, sigma: float = 1):
  keys = spin_to_number((sample + 1) // 2).flatten()
  return (sigma * gen_noise(keys)).reshape(sample.shape[:-1])

class NoisyRBM(RBM):
    sigma_noise: float = 1 

    @nn.compact
    def __call__(self, input):
        x = nn.Dense(
            name="Dense",
            features=int(self.alpha * input.shape[-1]),
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=self.use_hidden_bias,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
        )(input)
        x = self.activation(x)
        x = jnp.sum(x, axis=-1)

        if self.use_visible_bias:
            v_bias = self.param(
                "visible_bias",
                self.visible_bias_init,
                (input.shape[-1],),
                self.param_dtype,
            )
            out_bias = jnp.dot(input, v_bias)
            x = x + out_bias

        return x + sample_to_noise(input, self.sigma_noise)