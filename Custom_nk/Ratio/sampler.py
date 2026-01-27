from netket.sampler import MetropolisSampler
import jax
import jax.numpy as jnp
from netket.jax import apply_chunked
from functools import partial
from collections.abc import Callable
import jax
from jax import numpy as jnp
from flax import linen as nn
from netket.utils.types import PyTree
from netket.sampler.metropolis import _assert_good_sample_shape, _assert_good_log_prob_shape, SamplerState
from netket.utils import wrap_afun

class rSampler(MetropolisSampler):
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
                ratio = jnp.exp(proposal_log_prob - s["log_prob"] + log_prob_correction.astype(self.dtype))
            else:
                ratio = jnp.exp(proposal_log_prob - s["log_prob"])

            do_accept = uniform < ratio
            # do_accept must match ndim of proposal and state (which is 2)
            s["σ"] = jnp.where(do_accept.reshape(-1, 1), σp, s["σ"])
            s["accepted"] += do_accept

            s["log_prob"] = jax.numpy.where(
                do_accept.reshape(-1), proposal_log_prob, s["log_prob"]
            )
            s["ratio"] = ratio

            return s

        s = {
            "key": state.rng,
            "σ": state.σ,
            # Log prob is already computed in reset, so don't recompute it.
            # "log_prob": self.machine_pow * apply_machine(parameters, state.σ).real,
            "log_prob": state.log_prob,
            # for logging
            "accepted": state.n_accepted_proc,
            "ratio": jnp.ones_like(state.log_prob)
        }
        s = jax.lax.fori_loop(0, self.sweep_size, loop_body, s)

        new_state = state.replace(
            rng=s["key"],
            σ=s["σ"],
            log_prob=s["log_prob"],
            n_accepted_proc=s["accepted"],
            n_steps_proc=state.n_steps_proc + self.sweep_size * self.n_batches,
        )

        return new_state, (new_state.σ, new_state.log_prob, s["ratio"])
    
    @partial(
        jax.jit, static_argnames=("machine", "chain_length", "return_log_probabilities", "return_ratios")
    )
    def _sample_chain(
        self,
        machine,
        parameters,
        state,
        chain_length,
        return_log_probabilities: bool = False,
        return_ratios: bool = False
    ):
        """
        Samples `chain_length` batches of samples along the chains.

        Internal method used for jitting calls.

        Arguments:
            machine: A Flax module with the forward pass of the log-pdf.
            parameters: The PyTree of parameters of the model.
            state: The current state of the sampler.
            chain_length: The length of the chains.

        Returns:
            σ: The next batch of samples.
            state: The new state of the sampler
        """
        state, (samples, log_probabilities, ratios) = jax.lax.scan(
            lambda state, _: self._sample_next(machine, parameters, state),
            state,
            xs=None,
            length=chain_length,
        )
        # make it (n_chains, n_samples_per_chain) as expected by netket.stats.statistics
        samples = jnp.swapaxes(samples, 0, 1)
        log_probabilities = jnp.swapaxes(log_probabilities, 0, 1)
        ratios = jnp.swapaxes(ratios, 0, 1)

        if return_log_probabilities and return_ratios:
            return (samples, log_probabilities, ratios), state
        elif return_log_probabilities:
            return (samples, log_probabilities), state
        elif return_ratios:
            return (samples, ratios), state
        else:
            return samples, state
        
    def sample(
        self,
        machine: Callable | nn.Module,
        parameters: PyTree,
        *,
        state: SamplerState | None = None,
        chain_length: int = 1,
        return_log_probabilities: bool = False,
        return_ratios: bool = False
    ) -> (
        tuple[jax.Array, SamplerState]
        | tuple[tuple[jax.Array, jax.Array], SamplerState]
    ):
        """
        Samples `chain_length` batches of samples along the chains.

        Arguments:
            machine: A Flax module or callable with the forward pass of the log-pdf.
                If it is a callable, it should have the signature :code:`f(parameters, σ) -> jax.Array`.
            parameters: The PyTree of parameters of the model.
            state: The current state of the sampler. If not specified, then initialize and reset it.
            chain_length: The length of the chains (default = 1).
            return_log_probabilities: If `True`, the log-probabilities are also returned, which is sometimes
                useful to avoid re-evaluating the log-pdf when doing importance sampling. Defaults to False.

        Returns:
            Returns a tuple of 'samples' and 'state'. If `return_log_probabilities` is False,
            the samples are just the 3-rank array of samples. If `return_log_probabilities` is
            True, the samples are a tuple of the 3-rank array of samples and the 2-rank array of
            un-normalized log-probabilities corresponding to each sample.
        """
        if state is None:
            state = self.reset(machine, parameters)

        return self._sample_chain(
            wrap_afun(machine),
            parameters,
            state,
            chain_length,
            return_log_probabilities=return_log_probabilities,
            return_ratios=return_ratios
        )