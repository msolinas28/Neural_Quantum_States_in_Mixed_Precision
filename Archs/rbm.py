import numpy as np
import jax.numpy as jnp 
from flax import linen as nn
from netket.models.rbm import (
    RBM as nkRBM,
    RBMSymm as nkRBMSymm
)

def log_cosh(x, dtype):
    """
    Logarithm of the hyperbolic cosine, implemented in a more stable way.
    """ 
    scalar_log2 = jnp.log(2.0).astype(dtype)
    scalar_neg2 = jnp.array(-2.0, dtype=dtype)
    scalar_1 = jnp.array(1.0, dtype=dtype)

    sgn_x = scalar_neg2 * jnp.signbit(x.real).astype(dtype) + scalar_1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - scalar_log2

class RBM(nkRBM):
    def __init__(self, **kwargs):
        dtype = kwargs.get("param_dtype", jnp.float64)
        kwargs["activation"] = lambda x: log_cosh(x, dtype)
        super().__init__(**kwargs)
    
    @property
    def label(self):
        return f"RBM_alpha{self.alpha}"
            
class RBMSymm(nkRBMSymm):
    def __init__(self, **kwargs):
        dtype = kwargs.get("param_dtype", jnp.float64)
        kwargs["activation"] = lambda x: log_cosh(x, dtype)
        super().__init__(**kwargs)
    
    @property
    def label(self):
        return f"RBMSymm_alpha{self.alpha}"