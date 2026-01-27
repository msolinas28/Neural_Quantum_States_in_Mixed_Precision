import jax
import typing as tp
import jax.numpy as jnp
from jax import lax
from flax.typing import Array, Dtype, Axes
from typing import Any
from flax.linen.normalization import LayerNorm, _normalize, _canonicalize_axes, _abs_sq
from flax.linen import module

compact = module.compact

def _custom_compute_stats(
    x: Array,
    axes: Axes,
    dtype: Dtype | None,
    axis_name: str | None = None,
    axis_index_groups: Any = None,
    use_mean: bool = True,
    use_fast_variance: bool = True,
    mask: Array | None = None,
    force_float32_reductions = True,
    upcast_sums: bool = True
):
  
    if dtype is None:
        dtype = jnp.result_type(x)

    x = jnp.asarray(x, dtype)
    axes = _canonicalize_axes(x.ndim, axes)

    def maybe_distributed_mean(*xs, mask=None):
        mus = tuple(x.mean(axes, where=mask, dtype=None if upcast_sums else dtype) for x in xs)
        if axis_name is None:
            return mus if len(xs) > 1 else mus[0]
        else:
        # In the distributed case we stack multiple arrays to speed comms.
            if len(xs) > 1:
                reduced_mus = lax.pmean(
                jnp.stack(mus, axis=0),
                axis_name,
                axis_index_groups=axis_index_groups,
                )
                return tuple(reduced_mus[i] for i in range(len(xs)))
            else:
                return lax.pmean(mus[0], axis_name, axis_index_groups=axis_index_groups)

    if use_mean:
        if use_fast_variance:
            mu, mu2 = maybe_distributed_mean(x, _abs_sq(x), mask=mask)
        # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
        # to floating point round-off errors.
            var = jnp.maximum(0.0, mu2 - _abs_sq(mu))
        else:
            mu = maybe_distributed_mean(x, mask=mask)
            var = maybe_distributed_mean(
                _abs_sq(x - jnp.expand_dims(mu, axes)), mask=mask
            )
    else:
        var = maybe_distributed_mean(_abs_sq(x), mask=mask)
        mu = jnp.zeros_like(var)
    return mu, var

class CustomLayerNorm(LayerNorm):
    upcast_sums: bool = True

    @compact
    def __call__(self, x, *, mask: jax.Array | None = None):
        """Applies layer normalization on the input.

        Args:
        x: the inputs
        mask: Binary array of shape broadcastable to ``inputs`` tensor, indicating
            the positions for which the mean and variance should be computed.

        Returns:
        Normalized inputs (the same shape as inputs).
        """

        mean, var = _custom_compute_stats(
            x,
            self.reduction_axes,
            self.dtype,
            self.axis_name,
            self.axis_index_groups,
            use_fast_variance=self.use_fast_variance,
            mask=mask,
            force_float32_reductions=self.force_float32_reductions,
            upcast_sums=self.upcast_sums
        )

        return _normalize(
            self,
            x,
            mean,
            var,
            self.reduction_axes,
            self.feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
            self.force_float32_reductions,
        ) 