# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import numpy as np
import jax

from jax import numpy as jnp
from flax import linen as nn
from jax.nn.initializers import normal

from netket.utils.types import NNInitFunc

default_kernel_init = normal(stddev=0.01)

def log_cosh(x, dtype):
    """
    Logarithm of the hyperbolic cosine, implemented in a more stable way.
    """
    scalar_log2 = jnp.array(np.log(2.0), dtype=dtype)
    scalar_neg2 = jnp.array(-2.0, dtype=dtype)
    scalar_1 = jnp.array(1.0, dtype=dtype)

    sgn_x = scalar_neg2 * jnp.signbit(x.real).astype(dtype) + scalar_1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - scalar_log2


class RBM(nn.Module):
    r"""A restricted boltzman Machine, equivalent to a 2-layer FFNN with a
    nonlinear activation function in between.
    """
    param_dtype: Any = np.float64
    """The dtype of the weights."""
    alpha: float | int = 1
    """feature density. Number of features equal to alpha * input.shape[-1]"""
    use_hidden_bias: bool = True
    """if True uses a bias in the dense layer (hidden layer bias)."""
    use_visible_bias: bool = True
    """if True adds a bias to the input not passed through the nonlinear layer."""
    precision: Any = None
    """numerical precision of the computation see :class:`jax.lax.Precision` for details."""

    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    hidden_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the hidden bias."""
    visible_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the visible bias."""

    @nn.compact
    def __call__(self, input):
        is_integer_output = self.param_dtype in [
            jnp.int4,
            jnp.int8,
            jnp.int16,
            jnp.int32,
            jnp.int64,
        ]
        
        internal_param_dtype = jnp.float64 if is_integer_output else self.param_dtype

        x = nn.Dense(
            name="Dense",
            features=int(self.alpha * input.shape[-1]),
            param_dtype=internal_param_dtype,
            precision=self.precision,
            use_bias=self.use_hidden_bias,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
        )(input)

        x = log_cosh(x, internal_param_dtype)
        x = jnp.sum(x, axis=-1, dtype=internal_param_dtype)

        if self.use_visible_bias:
            v_bias = self.param(
                "visible_bias",
                self.visible_bias_init,
                (input.shape[-1],),
                internal_param_dtype,
            )
            out_bias = jnp.dot(input, v_bias)
            x = x + out_bias

        if is_integer_output:
            return x.astype(self.param_dtype)
        else:
            return x

    @property
    def label(self):
        return f"RBM_alpha{self.alpha}"