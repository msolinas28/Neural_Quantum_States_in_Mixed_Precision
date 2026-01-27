import typing as tp
import jax.numpy as jnp
import flax.linen as nn
from flax import nnx
from flax.linen import Module
from flax.nnx.nn import initializers
from netket.utils.types import NNInitFunc

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

from Archs.conv_2d import Conv2D
from Archs.layer_norm import CustomLayerNorm

default_kernel_init = initializers.he_normal()
default_bias_init = initializers.he_normal()

# If the input is an integer gelu will upcast to f32 and return in f32
default_activation = nnx.gelu

def _reshape(x, linear_size):
    if len(x.shape) > 1:
        x = x.reshape((x.shape[:-1] + (linear_size, linear_size, 1))) # (Batch_size, L, L, in_filters=1)
    else:
        x = x.reshape((linear_size, linear_size, 1)) # (L, L, in_filters=1)
    
    return x

class ResBlock(Module):
    linear_size: int
    filters: int = 1
    kernel_shape: tuple[int, int] = (1, 1)
    precision: tp.Any = None
    use_bias: bool = False
    param_dtype: any = jnp.float64
    kernel_init: NNInitFunc = default_kernel_init
    activation: tp.Any = default_activation
    upcast_sums: bool = True
    reshape: bool = True

    @nn.compact
    def __call__(self, input):
        if self.reshape: 
            input = _reshape(input, self.linear_size)

        x = CustomLayerNorm(
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
            upcast_sums=self.upcast_sums 
        )(input)

        x = self.activation(x)

        x = nn.Conv(
            features=self.filters,
            kernel_size=self.kernel_shape,
            use_bias=self.use_bias,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=self.kernel_init
        )(x)

        x = self.activation(x)

        x = nn.Conv(
            features=self.filters,
            kernel_size=self.kernel_shape,
            use_bias=self.use_bias,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=self.kernel_init
        )(x)

        return x + input
    
class ResCNN(Module):
    linear_size: int
    n_res_blocks: int = 1
    filters: int = 1
    kernel_shape: tuple[int, int] = (1, 1)
    precision: tp.Any = None
    use_bias: bool = False
    param_dtype: any = jnp.float64
    kernel_init: NNInitFunc = default_kernel_init
    activation: tp.Any = default_activation
    upcast_sums: bool = True
    reshape: bool = True

    @nn.compact
    def __call__(self, input):
        if self.reshape: 
            input = _reshape(input, self.linear_size)
            
        x = nn.Conv(
            features=self.filters,
            kernel_size=self.kernel_shape,
            use_bias=self.use_bias,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=self.kernel_init
        )(input)

        for _ in range(self.n_res_blocks):
            x = ResBlock(
                linear_size=self.linear_size,
                filters=self.filters,
                kernel_shape=self.kernel_shape,
                precision=self.precision,
                use_bias=self.use_bias,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                activation=self.activation,
                upcast_sums=self.upcast_sums,
                reshape=False
            )(x)
        
        x = CustomLayerNorm(
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
            upcast_sums=self.upcast_sums 
        )(x)

        return jnp.sum(x, axis=[-1, -2, -3], dtype=None if self.upcast_sums else self.param_dtype)

    @property
    def label(self):
        return f"ResCNN_nblocks{self.n_res_blocks}_nfilters{self.filters}_KernelShape{self.kernel_shape[0]}x{self.kernel_shape[1]}"