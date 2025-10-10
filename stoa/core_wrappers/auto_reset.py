from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from chex import PRNGKey

from stoa.core_wrappers.wrapper import Wrapper, WrapperState, wrapper_state_replace
from stoa.env_types import Action, EnvParams, Observation, State, TimeStep
from stoa.environment import Environment
from stoa.stoa_struct import dataclass

NEXT_OBS_KEY_IN_EXTRAS = "next_obs"


def add_obs_to_extras(timestep: TimeStep) -> TimeStep:
    """Place the observation in timestep.extras[NEXT_OBS_KEY_IN_EXTRAS].

    Used when auto-resetting to store the observation from the terminal TimeStep.
    This is particularly useful for algorithms that need access to the true final
    observation in truncated episodes.

    Args:
        timestep: TimeStep object containing the timestep returned by the environment.

    Returns:
        TimeStep with observation stored in extras[NEXT_OBS_KEY_IN_EXTRAS].
    """
    extras = {**timestep.extras, NEXT_OBS_KEY_IN_EXTRAS: timestep.observation}
    return timestep.replace(extras=extras)  # type: ignore


class AutoResetWrapper(Wrapper[State]):
    """Automatically resets the environment when episodes terminate.

    This wrapper intercepts terminal timesteps and automatically calls reset(),
    replacing the terminal observation with the initial observation of the new episode.
    Optionally preserves the original terminal observation in timestep.extras.

    The wrapper expects the environment state to have a 'rng_key' attribute that provides
    the source of randomness for automatic resets.
    """

    def __init__(self, env: Environment, keep_terminal: bool = False):
        """Initialize the AutoResetWrapper.

        Args:
            env: The environment to wrap.
            keep_terminal: If True, return terminal observations.
                Otherwise, store the terminal observation in `timestep.extras[NEXT_OBS_KEY_IN_EXTRAS]`.
        """
        super().__init__(env)
        self.keep_terminal = keep_terminal

    def reset(
        self, rng_key: PRNGKey, env_params: EnvParams | None = None
    ) -> tuple[State, TimeStep]:
        """Reset the environment.

        Args:
            rng_key: Random key for environment reset.
            env_params: Optional environment parameters.

        Returns:
            Tuple of (initial_state, initial_timestep).
            If self.keep_terminal is True, the initial state will be a tuple
            where the second element indicates whether the state is terminal.
            Otherwise, the initial_timestep will have the observation added to extras.
        """
        state, timestep = self._env.reset(rng_key, env_params)
        if self.keep_terminal:
            state = (state, jnp.asarray(False))
        else:
            timestep = add_obs_to_extras(timestep)
        return state, timestep

    def step(
        self, state: State, action: Action, env_params: EnvParams | None = None
    ) -> tuple[State, TimeStep]:
        """Step the environment with automatic resetting on termination or truncation.

        If the episode terminates after the step, the environment is automatically
        reset and the terminal observation is replaced with the reset observation.
        The terminal reward and done flag are preserved in the timestep.

        Args:
            state: Current environment state.
            action: Action to take.
            env_params: Optional environment parameters.

        Returns:
            Tuple of (new_state, new_timestep). If the episode terminated:
            - new_state corresponds to the automatically reset environment
            - new_timestep preserves the terminal reward and done flag but
              contains the reset observation
            If the episode didn't terminate:
            - new_state is the stepped environment state
            - new_timestep is the stepped timestep, optionally with observation
              added to extras
        """
        if self.keep_terminal:
            state, done = state

            def auto_reset(state, _, env_params):
                return self.reset(state.rng_key, env_params)

            def no_reset(state, action, env_params):
                step_state, step_timestep = self._env.step(state, action, env_params)
                return (step_state, step_timestep.done()), step_timestep

            return jax.lax.cond(done, auto_reset, no_reset, state, action, env_params)
        else:
            state, timestep = self._env.step(state, action, env_params)
            timestep = add_obs_to_extras(timestep)

            def auto_reset(state, timestep, env_params):
                # only keep observation from reset_timestep
                reset_state, reset_timestep = self.reset(state.rng_key, env_params)
                return reset_state, timestep.replace(observation=reset_timestep.observation)

            return jax.lax.cond(
                timestep.done(), auto_reset, lambda *args: args[:2], state, timestep, env_params
            )


@dataclass(custom_replace_fn=wrapper_state_replace)
class CachedAutoResetState(WrapperState):
    """State for cached auto-reset wrapper."""

    cached_state: State
    cached_timestep: TimeStep


class CachedAutoResetWrapper(Wrapper[CachedAutoResetState]):
    """Auto-reset wrapper that caches the initial reset for repeated use."""

    def __init__(self, env: Environment, keep_terminal: bool = False):
        super().__init__(env)
        self.keep_terminal = keep_terminal

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[CachedAutoResetState, TimeStep]:
        """Reset and cache the initial state and observation."""
        env_state, timestep = self._env.reset(rng_key, env_params)
        state = CachedAutoResetState(env_state, env_state, timestep)
        if self.keep_terminal:
            state = (state, jnp.asarray(False))
        else:
            timestep = add_obs_to_extras(timestep)

        return state, timestep

    def step(
        self,
        state: CachedAutoResetState,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[CachedAutoResetState, TimeStep]:
        """Step with cached auto-reset on episode termination."""

        if self.keep_terminal:
            state, done = state

            def auto_reset(state, action, env_params):
                state = state.replace(base_env_state=state.cached_state)
                return (state, jnp.asarray(False)), state.cached_timestep

            def no_reset(state, action, env_params):
                env_state, step_timestep = self._env.step(state.base_env_state, action, env_params)
                state = state.replace(base_env_state=env_state)
                return (state, step_timestep.done()), step_timestep

            return jax.lax.cond(done, auto_reset, no_reset, state, action, env_params)

        else:
            env_state, timestep = self._env.step(state.base_env_state, action, env_params)
            state = state.replace(base_env_state=env_state)
            timestep = add_obs_to_extras(timestep)

            def auto_reset(state, timestep):
                state = state.replace(base_env_state=state.cached_state)
                timestep = timestep.replace(observation=state.cached_timestep.observation)
                return state, timestep

            return jax.lax.cond(timestep.done(), auto_reset, lambda *args: args, state, timestep)
