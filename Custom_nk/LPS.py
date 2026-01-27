from netket.sampler import MetropolisSampler
import jax
import jax.numpy as jnp
from netket.jax import apply_chunked
from netket.sampler.metropolis import _assert_good_log_prob_shape, _assert_good_sample_shape
from netket.utils import struct
from netket.hilbert import AbstractHilbert
from netket.sampler.rules import MetropolisRule
from netket.utils.types import DType

class Custom_sampler(MetropolisSampler):
    dtype_ratio: any = struct.field(pytree_node=False)

    def __init__(
        self,
        hilbert: AbstractHilbert,
        rule: MetropolisRule,
        *,
        n_sweeps: int = None,
        sweep_size: int = None,
        reset_chains: bool = False,
        n_chains: int | None = None,
        n_chains_per_rank: int | None = None,
        chunk_size: int | None = None,
        machine_pow: int = 2,
        dtype: DType = None,
        dtype_ratio = None
    ):
        super().__init__(
            hilbert,
            rule,
            n_sweeps = n_sweeps,
            sweep_size = sweep_size,
            reset_chains = reset_chains,
            n_chains = n_chains,
            n_chains_per_rank = n_chains_per_rank,
            chunk_size = chunk_size,
            machine_pow = machine_pow,
            dtype = dtype
        )
        self.dtype_ratio = dtype_ratio if dtype_ratio is not None else self.dtype

    def _sample_next(self, machine, parameters, state):
        """
        Implementation of `sample_next` for subclasses of `MetropolisSampler`.

        If you subclass `MetropolisSampler`, you should override this and not `sample_next`
        itself, because `sample_next` contains some common logic.
        """
        apply_machine = apply_chunked(
            machine.apply, in_axes=(None, 0), chunk_size=self.chunk_size
        )

        def loop_body(i, s):
            # 1 to propagate for next iteration, 1 for uniform rng and n_chains for transition kernel
            s["key"], key1, key2 = jax.random.split(s["key"], 3)

            σp, log_prob_correction = self.rule.transition(
                self, machine, parameters, state, key1, s["σ"]
            )
            _assert_good_sample_shape(
                σp,
                (self.n_batches, self.hilbert.size),
                self.dtype,
                f"{self.rule}.transition",
            )
            proposal_log_prob = self.machine_pow * apply_machine(parameters, σp).real
            _assert_good_log_prob_shape(proposal_log_prob, self.n_batches, machine)

            uniform = jax.random.uniform(key2, shape=(self.n_batches,))
            if log_prob_correction is not None:
                do_accept = uniform < jnp.exp(
                    proposal_log_prob - s["log_prob"] + log_prob_correction.astype(self.dtype_ratio)
                )
            else:
                do_accept = uniform < jnp.exp(proposal_log_prob - s["log_prob"])

            # do_accept must match ndim of proposal and state (which is 2)
            s["σ"] = jnp.where(do_accept.reshape(-1, 1), σp, s["σ"])
            s["accepted"] += do_accept

            s["log_prob"] = jax.numpy.where(
                do_accept.reshape(-1), proposal_log_prob, s["log_prob"]
            )

            return s

        s = {
            "key": state.rng,
            "σ": state.σ,
            # Log prob is already computed in reset, so don't recompute it.
            # "log_prob": self.machine_pow * apply_machine(parameters, state.σ).real,
            "log_prob": state.log_prob,
            # for logging
            "accepted": state.n_accepted_proc,
        }
        s = jax.lax.fori_loop(0, self.sweep_size, loop_body, s)

        new_state = state.replace(
            rng=s["key"],
            σ=s["σ"],
            log_prob=s["log_prob"],
            n_accepted_proc=s["accepted"],
            n_steps_proc=state.n_steps_proc + self.sweep_size * self.n_batches,
        )

        return new_state, (new_state.σ, new_state.log_prob)