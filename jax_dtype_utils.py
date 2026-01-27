import jax.numpy as jnp

def str_to_jax_dtype(srt_dtype):
    mapping = {
        "f64": jnp.float64,
        "f32": jnp.float32,
        "f16": jnp.float16,
        "bf16": jnp.bfloat16,
    }
    try:
        return mapping[srt_dtype.lower()]
    except KeyError:
        raise NotImplementedError("Pick a valid dtype: 'f64', 'f32', 'f16', or 'bf16'")
    
def jax_dtype_to_str(jax_dtype):
    mapping = {
        "f64": jnp.float64,
        "f32": jnp.float32,
        "f16": jnp.float16,
        "bf16": jnp.bfloat16,
    }

    for key, value in mapping.items():
        if value == jax_dtype:
            return key
    raise NotImplementedError("Pick a valid jax dtype: jnp.float64, jnp.float32, jnp.float16 or jnp.floatb16")

def is_jax_dtype(x):
    return x in [jnp.float64, jnp.float32, jnp.float16, jnp.bfloat16]

class jax_dtype():
    def __init__(self, dtype):
        if dtype is None or (isinstance(dtype, str) and dtype.lower() == "none"):
            self.label = None
            self.dtype = None
        elif isinstance(dtype, str):
            self.label = dtype
            self.dtype = str_to_jax_dtype(dtype)
        elif is_jax_dtype(dtype):
            self.label = jax_dtype_to_str(dtype)
            self.dtype = dtype
        else:
            raise NotImplementedError("dtype has to be a sting or a jax dtype.")