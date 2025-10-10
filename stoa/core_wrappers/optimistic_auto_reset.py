from functools import partial

import jax
import jax.numpy as jnp
from chex import PRNGKey

from stoa.core_wrappers.auto_reset import add_obs_to_extras
from stoa.core_wrappers.wrapper import Wrapper
from stoa.env_types import Action, EnvParams, State, TimeStep
from stoa.environment import Environment


class OptimisticResetVmapWrapper(Wrapper[State]):
    """
    Efficient vectorized environment wrapper with optimistic resets.

    This wrapper combines environment vectorization (vmap) with automatic resets,
    using an "optimistic" strategy that pre-generates reset states for efficiency.
    Instead of resetting environments one-by-one as they terminate, it generates
    a smaller number of reset states and distributes them to terminated environments.

    Note: This wrapper requires the environment state to have an 'rng_key' attribute.
    Use the AddRNGKey wrapper before this one if your environment doesn't have it.

    Args:
        env: The base environment to vectorize.
        num_envs: Number of parallel environments to run.
        reset_ratio: Number of environments per reset state generated.
                    Higher values are more efficient but may cause duplicate resets.
                    Must divide num_envs evenly.
    """

    def __init__(
        self, env: Environment, num_envs: int, reset_ratio: int, keep_terminal: bool = False
    ):
        super().__init__(env)
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if num_envs % reset_ratio != 0:
            raise ValueError(
                f"reset_ratio ({reset_ratio}) must evenly divide num_envs ({num_envs})"
            )

        self.num_envs = num_envs
        self.num_resets = num_envs // reset_ratio
        self._vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
        self._vmap_step = jax.vmap(env.step, in_axes=(0, 0, None))
        self.keep_terminal = keep_terminal

    def reset(
        self, rng_key: PRNGKey, env_params: EnvParams | None = None
    ) -> tuple[State, TimeStep]:
        if rng_key.ndim == 1 if rng_key.dtype == jnp.uint32 else rng_key.ndim == 0:
            rng_key = jax.random.split(rng_key, self.num_envs)
        states, timesteps = self._vmap_reset(rng_key, env_params)
        if self.keep_terminal:
            states = (states, jnp.zeros_like(rng_key, dtype=jnp.bool_))
        else:
            timesteps = add_obs_to_extras(timesteps)
        return states, timesteps

    def step(
        self, state: State, action: Action, env_params: EnvParams | None = None
    ) -> tuple[State, TimeStep]:
        if self.keep_terminal:
            env_states, dones = state
        else:
            env_states = state

        # get the reset states and stepped states
        reset_keys, state_keys = jax.vmap(jax.random.split, out_axes=1)(env_states.rng_key)
        env_states = env_states.replace(rng_key=state_keys)
        step_states, step_timesteps = self._vmap_step(env_states, action, env_params)
        env_to_reset_index = jax.random.choice(
            shape=(self.num_envs,), a=self.num_resets, key=reset_keys[self.num_resets]
        )
        reset_states, reset_timesteps = jax.tree.map(
            lambda x: x[env_to_reset_index],
            self.reset(reset_keys[: self.num_resets], env_params),
        )

        if self.keep_terminal:
            step_states = (step_states, step_timesteps.done())
        else:
            step_timesteps = add_obs_to_extras(step_timesteps)
            dones = step_timesteps.done()
            reset_timesteps = step_timesteps.replace(observation=reset_timesteps.observation)

        tree_where = lambda done, x, y: jax.tree.map(partial(jnp.where, done), x, y)
        states = jax.vmap(tree_where)(dones, reset_states, step_states)
        timesteps = jax.vmap(tree_where)(dones, reset_timesteps, step_timesteps)

        return states, timesteps
