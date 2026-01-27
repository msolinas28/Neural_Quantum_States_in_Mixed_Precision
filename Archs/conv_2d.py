import jax.numpy as jnp
import flax.linen as nn
import typing as tp
from flax.linen import initializers
from flax.core import Scope
from netket.utils.types import NNInitFunc
import jax.random as jr
import numpy as np

class Conv2D(nn.Module):
    linear_size: int
    filters: int = 1
    kernel_shape: tuple[int, int] = (1, 1)
    precision: tp.Any = None
    use_bias: bool = False
    param_dtype: any = jnp.float64
    kernel_init: NNInitFunc = initializers.lecun_normal()
    reshape: bool = True

    @nn.compact
    def __call__(self, x):
        if self.reshape:
            x = x.reshape((x.shape[:-1] + (self.linear_size, self.linear_size, 1))) # (Batch_size, L, L, in_filters=1)
        
        x = nn.Conv(
            features=self.filters,
            kernel_size=self.kernel_shape,
            use_bias=self.use_bias,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=self.kernel_init
        )(x)

        return x
