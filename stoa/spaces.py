"""Spaces for stoa."""

import abc
from functools import cached_property, partial
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import jax
import jax.numpy as jnp
import numpy as np
from chex import PRNGKey, Shape
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from stoa.env_types import Action, StepType, TimeStep

T = TypeVar("T")


class Space(abc.ABC, Generic[T]):
    """Abstract base class for spaces that describe RL environment domains.

    Spaces define the valid structure and bounds for observations, actions,
    rewards, and other components of RL environments.
    """

    @cached_property
    @abc.abstractmethod
    def shape(self) -> Optional[Shape]:
        """The shape of values described by this space.

        Returns None for composite spaces (Dict, Tuple).
        """
        pass

    @cached_property
    @abc.abstractmethod
    def dtype(self) -> Optional[jnp.dtype]:
        """The dtype of values described by this space.

        Returns None for composite spaces (Dict, Tuple).
        """
        pass

    @abc.abstractmethod
    def sample(self, rng_key: PRNGKey) -> T:
        """Sample a random value from this space.

        Args:
            rng_key: A JAX PRNG key.

        Returns:
            A value sampled from this space.
        """
        pass

    @abc.abstractmethod
    def contains(self, value: Any) -> Array:
        """Checks if a value conforms to this space.

        Args:
            value: The value to check.

        Returns:
            A boolean indicating whether the value conforms to the space.
        """
        pass

    def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
        """Flatten this space for JAX tree operations."""
        return [], self.__dict__

    @classmethod
    def tree_unflatten(cls, aux_data: Dict[str, Any], children: Sequence[Any]) -> "Space":
        """Unflatten this space from JAX tree operations."""
        return cls(**aux_data)

    def generate_value(self) -> T:
        """Generate a dummy value from this space."""
        return self.sample(jax.random.PRNGKey(0))

    @cached_property
    def name(self) -> str:
        """Get a semantic name for this space."""
        return self.__class__.__name__


class ArraySpace(Space[Array]):
    """Describes a JAX array with a specific shape and dtype."""

    def __init__(
        self,
        shape: Union[int, Iterable[int]],
        dtype: DTypeLike = float,
        name: str = "",
    ) -> None:
        """Initializes a new `ArraySpace`.

        Args:
            shape: An integer or iterable specifying the array shape.
            dtype: JAX numpy dtype or type specifying the array dtype.
            name: Optional string containing a semantic name for the array.
        """
        if isinstance(shape, int):
            shape = (shape,)
        self._shape = tuple(int(dim) for dim in shape)
        self._dtype = jnp.dtype(dtype)
        self._name = name

    def __repr__(self) -> str:
        return f"ArraySpace(shape={self.shape}, dtype={self.dtype}, name={self._name!r})"

    def __eq__(self, other: Any) -> bool:
        """Check equality with another space."""
        if not isinstance(other, ArraySpace):
            return False
        return (
            self._shape == other._shape
            and self._dtype == other._dtype
            and self._name == other._name
        )

    @cached_property
    def shape(self) -> Shape:
        """The shape of values described by this space."""
        return self._shape

    @cached_property
    def dtype(self) -> jnp.dtype:
        """The dtype of values described by this space."""
        return self._dtype

    @cached_property
    def name(self) -> str:
        """The name of this space."""
        return self._name

    @cached_property
    def size(self) -> int:
        """Total number of elements in arrays from this space."""
        return int(np.prod(self._shape))

    def sample(self, rng_key: PRNGKey) -> Array:
        """Sample a random value from this space.

        For an unbounded Array, samples from a standard normal distribution.

        Args:
            rng_key: A JAX PRNG key.

        Returns:
            A JAX array containing the sampled value.
        """
        # For a generic Array, sample from standard normal as it's unbounded
        return jax.random.normal(rng_key, shape=self.shape, dtype=self.dtype)

    def contains(self, value: Any) -> Array:
        """Checks if value conforms to this space.

        Args:
            value: A value to check.

        Returns:
            A boolean indicating whether the value conforms to the space.
        """
        if not isinstance(value, (np.ndarray, jax.Array)):
            return jnp.array(False)
        return jnp.array(value.shape == self.shape and value.dtype == self.dtype)

    def zeros(self) -> Array:
        """Create a zero array conforming to this space."""
        return jnp.zeros(self.shape, dtype=self.dtype)

    def ones(self) -> Array:
        """Create an array of ones conforming to this space."""
        return jnp.ones(self.shape, dtype=self.dtype)

    def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
        return [], {"shape": self._shape, "dtype": self._dtype, "name": self._name}

    @classmethod
    def tree_unflatten(cls, aux_data: Dict[str, Any], children: Sequence[Any]) -> "ArraySpace":
        return cls(**aux_data)


class BoundedArraySpace(ArraySpace):
    """Describes a JAX array with minimum and maximum bounds."""

    def __init__(
        self,
        shape: Union[int, Iterable[int]],
        dtype: DTypeLike = float,
        minimum: Union[float, int, ArrayLike] = -jnp.inf,
        maximum: Union[float, int, ArrayLike] = jnp.inf,
        name: str = "",
    ) -> None:
        """Initializes a new `BoundedArraySpace`.

        Args:
            shape: An integer or iterable specifying the array shape.
            dtype: JAX numpy dtype or type specifying the array dtype.
            minimum: Minimum values (inclusive). Must be broadcastable to `shape`.
            maximum: Maximum values (inclusive). Must be broadcastable to `shape`.
            name: Optional string containing a semantic name for the array.

        Raises:
            ValueError: If `minimum` or `maximum` are not broadcastable to `shape`.
            ValueError: If any values in `minimum` are greater than the corresponding
                value in `maximum`.
        """
        super().__init__(shape, dtype, name)

        # Convert to JAX arrays
        self._minimum = jnp.asarray(minimum, dtype=self._dtype)
        self._maximum = jnp.asarray(maximum, dtype=self._dtype)

        # Check that minimum and maximum are broadcastable to shape
        try:
            self._minimum_broadcast = jnp.broadcast_to(self._minimum, shape=self._shape)
        except ValueError as e:
            raise ValueError(f"`minimum` is not broadcastable to shape {self._shape}") from e

        try:
            self._maximum_broadcast = jnp.broadcast_to(self._maximum, shape=self._shape)
        except ValueError as e:
            raise ValueError(f"`maximum` is not broadcastable to shape {self._shape}") from e

        # Check that all minimums are <= maximums
        if jnp.any(self._minimum_broadcast > self._maximum_broadcast):
            raise ValueError(
                f"All values in `minimum` must be <= the corresponding value in `maximum`.\n"
                f"Got minimum={self._minimum} and maximum={self._maximum}"
            )

    def __repr__(self) -> str:
        return (
            f"BoundedArraySpace(shape={self.shape}, dtype={self.dtype}, "
            f"minimum={self.minimum}, maximum={self.maximum}, name={self.name!r})"
        )

    def __eq__(self, other: Any) -> bool:
        """Check equality with another space."""
        if not isinstance(other, BoundedArraySpace):
            return False
        return (
            super().__eq__(other)
            and jnp.array_equal(self._minimum, other._minimum)
            and jnp.array_equal(self._maximum, other._maximum)
        )

    @cached_property
    def minimum(self) -> Array:
        """Minimum values (inclusive)."""
        return self._minimum

    @cached_property
    def maximum(self) -> Array:
        """Maximum values (inclusive)."""
        return self._maximum

    @cached_property
    def is_bounded(self) -> Array:
        """Whether this space has finite bounds."""
        return jnp.logical_and(
            jnp.all(jnp.isfinite(self._minimum_broadcast)),
            jnp.all(jnp.isfinite(self._maximum_broadcast)),
        )

    def sample(self, rng_key: PRNGKey) -> Array:
        """Sample a random value from this space within the bounds.

        Args:
            rng_key: A JAX PRNG key.

        Returns:
            A JAX array containing the sampled value.
        """
        bounded_sample_fn = partial(
            jax.random.uniform,
            shape=self.shape,
            minval=self._minimum_broadcast,
            maxval=self._maximum_broadcast,
        )
        unbounded_sample_fn = partial(
            jax.random.truncated_normal,
            shape=self.shape,
            lower=self._minimum_broadcast,
            upper=self._maximum_broadcast,
        )
        sample = jax.lax.cond(
            self.is_bounded, bounded_sample_fn, unbounded_sample_fn, rng_key
        ).astype(self.dtype)
        return sample

    def contains(self, value: Any) -> Array:
        """Checks if value conforms to this space.

        Args:
            value: A value to check.

        Returns:
            A boolean indicating whether the value conforms to the space.
        """
        # First check if the shape and dtype match
        if not super().contains(value):
            return jnp.array(False)

        # Check if value is within bounds
        return jnp.logical_and(
            jnp.all(value >= self._minimum_broadcast),
            jnp.all(value <= self._maximum_broadcast),
        )

    def clip(self, value: ArrayLike) -> Array:
        """Clip a value to the bounds of this space."""
        return jnp.clip(value, self._minimum_broadcast, self._maximum_broadcast)


class DiscreteSpace(BoundedArraySpace):
    """Describes a discrete scalar space with values from 0 to num_values-1."""

    def __init__(self, num_values: int, dtype: DTypeLike = int, name: str = "", shape=()) -> None:
        """Initializes a new `DiscreteSpace`.

        Args:
            num_values: Number of values in the space.
            dtype: JAX numpy dtype. Must be an integer type.
            name: Optional string containing a semantic name for the array.

        Raises:
            ValueError: If `num_values` is not positive.
            ValueError: If `dtype` is not an integer type.
        """
        if num_values <= 0:
            raise ValueError(f"`num_values` must be positive, got {num_values}")

        if not jnp.issubdtype(dtype, jnp.integer):
            raise ValueError(f"`dtype` must be an integer type, got {dtype}")

        super().__init__(
            shape=shape,  # Discrete spaces are scalar
            dtype=dtype,
            minimum=0,
            maximum=num_values - 1,
            name=name,
        )
        self._num_values = num_values

    def __repr__(self) -> str:
        return f"DiscreteSpace(num_values={self.num_values}, shape={self.shape}, dtype={self.dtype}, name={self.name!r})"

    def __eq__(self, other: Any) -> bool:
        """Check equality with another space."""
        if not isinstance(other, DiscreteSpace):
            return False
        return self._num_values == other._num_values and self._dtype == other._dtype

    @cached_property
    def num_values(self) -> int:
        """Number of possible discrete values."""
        return self._num_values

    def sample(self, rng_key: PRNGKey) -> Array:
        """Sample a random value from this space.

        Args:
            rng_key: A JAX PRNG key.

        Returns:
            A JAX array containing the sampled value.
        """
        return jax.random.randint(
            rng_key,
            shape=self.shape,
            minval=0,
            maxval=self.num_values,
            dtype=self.dtype,
        )

    def one_hot(self, value: ArrayLike) -> Array:
        """Convert a discrete value to one-hot encoding."""
        return jax.nn.one_hot(value, self.num_values, dtype=jnp.float32)

    def contains(self, value: Any) -> Array:
        """Checks if value conforms to this discrete space.

        Accepts Python ints, floats, NumPy scalars, and JAX scalars.
        """
        arr = jnp.asarray(value)
        if arr.shape != () or not jnp.issubdtype(arr.dtype, jnp.integer):
            return jnp.array(False)
        return super().contains(arr)


class MultiDiscreteSpace(BoundedArraySpace):
    """Describes a multi-dimensional discrete space.

    Each dimension has its own number of possible values.
    """

    def __init__(
        self,
        num_values: Union[Sequence[int], ArrayLike],
        dtype: DTypeLike = int,
        name: str = "",
    ) -> None:
        """Initializes a new `MultiDiscreteSpace`.

        Args:
            num_values: Number of values for each dimension.
            dtype: JAX numpy dtype. Must be an integer type.
            name: Optional string containing a semantic name for the array.

        Raises:
            ValueError: If any values in `num_values` are not positive.
            ValueError: If `dtype` is not an integer type.
        """
        self._num_values = jnp.asarray(num_values, dtype=jnp.int32)

        if jnp.any(self._num_values <= 0):
            raise ValueError(f"All values in `num_values` must be positive, got {num_values}")

        if not jnp.issubdtype(dtype, jnp.integer):
            raise ValueError(f"`dtype` must be an integer type, got {dtype}")

        super().__init__(
            shape=self._num_values.shape,
            dtype=dtype,
            minimum=jnp.zeros_like(self._num_values),
            maximum=self._num_values - 1,
            name=name,
        )

    def __repr__(self) -> str:
        return (
            f"MultiDiscreteSpace(num_values={list(self.num_values)}, "
            f"dtype={self.dtype}, name={self.name!r})"
        )

    def __eq__(self, other: Any) -> bool:
        """Check equality with another space."""
        if not isinstance(other, MultiDiscreteSpace):
            return False
        return jnp.array_equal(self._num_values, other._num_values) and self._dtype == other._dtype

    @cached_property
    def num_values(self) -> Array:
        """Number of possible values for each dimension."""
        return self._num_values

    def sample(self, rng_key: PRNGKey) -> Array:
        """Sample a random value from this space.

        Args:
            rng_key: A JAX PRNG key.

        Returns:
            A JAX array containing the sampled value.
        """
        # Vectorized sampling for all dimensions at once
        uniform_samples = jax.random.uniform(rng_key, shape=self.shape)
        return jnp.floor(uniform_samples * self.num_values).astype(self.dtype)


class DictSpace(Space[Dict[str, Any]]):
    """A dictionary of spaces."""

    def __init__(self, spaces: Dict[str, Space], name: str = "") -> None:
        """Initializes a new `DictSpace`.

        Args:
            spaces: A dictionary mapping keys to spaces.
            name: Optional string containing a semantic name for the dictionary.
        """
        self._spaces = dict(spaces)
        self._name = name

    def __repr__(self) -> str:
        spaces_str = ", ".join(f"{k}={v}" for k, v in self._spaces.items())
        return f"DictSpace({{{spaces_str}}}, name={self._name!r})"

    def __eq__(self, other: Any) -> bool:
        """Check equality with another space."""
        if not isinstance(other, DictSpace):
            return False
        return self._spaces == other._spaces and self._name == other._name

    @cached_property
    def name(self) -> str:
        """The name of this space."""
        return self._name

    @cached_property
    def shape(self) -> None:
        """Dict spaces don't have a shape."""
        return None

    @cached_property
    def dtype(self) -> None:
        """Dict spaces don't have a dtype."""
        return None

    @cached_property
    def spaces(self) -> Dict[str, Space]:
        """The constituent spaces."""
        return self._spaces.copy()

    def sample(self, rng_key: PRNGKey) -> Dict[str, Any]:
        """Sample a random value from this space.

        Args:
            rng_key: A JAX PRNG key.

        Returns:
            A dictionary mapping keys to sampled values.
        """
        keys = jax.random.split(rng_key, len(self._spaces))
        return {key: space.sample(keys[i]) for i, (key, space) in enumerate(self._spaces.items())}

    def contains(self, value: Dict[str, Any]) -> Array:
        """Checks if value conforms to this space.

        Args:
            value: A value to check.

        Returns:
            A boolean indicating whether the value conforms to the space.
        """
        if not isinstance(value, dict):
            return jnp.array(False)
        return jnp.array(all(space.contains(value[key]) for key, space in self._spaces.items()))

    def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
        return [], {"spaces": self._spaces, "name": self._name}

    @classmethod
    def tree_unflatten(cls, aux_data: Dict[str, Any], children: Sequence[Any]) -> "DictSpace":
        return cls(**aux_data)


class TupleSpace(Space[Tuple[Any, ...]]):
    """A tuple of spaces."""

    def __init__(self, spaces: Sequence[Space], name: str = "") -> None:
        """Initializes a new `TupleSpace`.

        Args:
            spaces: A sequence of spaces.
            name: Optional string containing a semantic name for the tuple.
        """
        self._spaces = tuple(spaces)
        self._name = name

    def __repr__(self) -> str:
        spaces_str = ", ".join(repr(space) for space in self._spaces)
        return f"TupleSpace(({spaces_str}), name={self._name!r})"

    def __eq__(self, other: Any) -> bool:
        """Check equality with another space."""
        if not isinstance(other, TupleSpace):
            return False
        return self._spaces == other._spaces and self._name == other._name

    def __len__(self) -> int:
        """Return the number of spaces in the tuple."""
        return len(self._spaces)

    @cached_property
    def name(self) -> str:
        """The name of this space."""
        return self._name

    @cached_property
    def shape(self) -> None:
        """Tuple spaces don't have a shape."""
        return None

    @cached_property
    def dtype(self) -> None:
        """Tuple spaces don't have a dtype."""
        return None

    @cached_property
    def spaces(self) -> Tuple[Space, ...]:
        """The constituent spaces."""
        return self._spaces

    def sample(self, rng_key: PRNGKey) -> Tuple[Any, ...]:
        """Sample a random value from this space.

        Args:
            rng_key: A JAX PRNG key.

        Returns:
            A tuple of sampled values.
        """
        keys = jax.random.split(rng_key, len(self._spaces))
        return tuple(space.sample(keys[i]) for i, space in enumerate(self._spaces))

    def contains(self, value: Any) -> Array:
        """Checks if value conforms to this space.

        Args:
            value: A value to check.

        Returns:
            A boolean indicating whether the value conforms to the space.
        """
        if not isinstance(value, tuple):
            return jnp.array(False)

        if len(value) != len(self._spaces):
            return jnp.array(False)

        # Check each value against its corresponding space
        return jnp.array(all(space.contains(val) for space, val in zip(self._spaces, value)))

    def __getitem__(self, index: int) -> Space:
        """Gets the space at an index."""
        return self._spaces[index]

    def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
        return [], {"spaces": self._spaces, "name": self._name}

    @classmethod
    def tree_unflatten(cls, aux_data: Dict[str, Any], children: Sequence[Any]) -> "TupleSpace":
        return cls(**aux_data)


class EnvironmentSpace(NamedTuple):
    """Full specification of the domains (spaces) used by a given environment.

    This class groups together all the spaces that define a complete RL environment.
    """

    observations: Space
    actions: Space
    rewards: Space
    discounts: Space
    state: Space

    def replace(self, **kwargs: Any) -> "EnvironmentSpace":
        """Returns a new `EnvironmentSpace` with the specified fields replaced."""
        return self._replace(**kwargs)

    def contains(self, timestep: TimeStep, action: Optional[Action] = None) -> Array:
        """Checks if a timestep and optional action conform to the space.

        Args:
            timestep: A timestep to validate. Should have observation, reward,
                     and discount fields.
            action: An optional action to validate.

        Returns:
            A boolean indicating whether the timestep and action conform to the space.
        """
        # Check timestep components
        obs_valid = self.observations.contains(timestep.observation)
        rew_valid = self.rewards.contains(timestep.reward)
        disc_valid = self.discounts.contains(timestep.discount)

        # Combine the validations for timestep components
        timestep_valid = jnp.logical_and(jnp.logical_and(obs_valid, rew_valid), disc_valid)

        if action is not None:
            act_valid = self.actions.contains(action)
            return jnp.logical_and(timestep_valid, act_valid)
        else:
            return timestep_valid

    def sample_timestep(self, rng_key: PRNGKey) -> TimeStep:
        """Sample a random timestep from this environment space.

        Args:
            rng_key: A JAX PRNG key.

        Returns:
            A timestep with observation, reward, and discount.
        """
        obs_key, rew_key, disc_key = jax.random.split(rng_key, 3)
        step_type = jax.random.choice(
            rng_key,
            jnp.array(
                [
                    StepType.FIRST,
                    StepType.MID,
                    StepType.TERMINATED,
                    StepType.TRUNCATED,
                ]
            ),
        )
        timestep = TimeStep(
            step_type=step_type,
            reward=self.rewards.sample(rew_key),
            discount=self.discounts.sample(disc_key),
            observation=self.observations.sample(obs_key),
        )
        return timestep

    def sample_transition(self, rng_key: PRNGKey) -> Dict[str, Any]:
        """Sample a random transition from this environment space.

        Args:
            rng_key: A JAX PRNG key.

        Returns:
            A dictionary with observation, action, reward, and discount.
        """
        obs_key, act_key, rew_key, disc_key = jax.random.split(rng_key, 4)
        return {
            "observation": self.observations.sample(obs_key),
            "action": self.actions.sample(act_key),
            "reward": self.rewards.sample(rew_key),
            "discount": self.discounts.sample(disc_key),
        }

    def sample_state(self, rng_key: PRNGKey) -> Any:
        """Sample a random state from this environment space.

        Args:
            rng_key: A JAX PRNG key.

        Returns:
            A state from this environment space.
        """
        return self.state.sample(rng_key)

    def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:
        """Flatten this space for JAX tree operations."""
        return [], self.__dict__

    @classmethod
    def tree_unflatten(
        cls, aux_data: Dict[str, Any], children: Sequence[Any]
    ) -> "EnvironmentSpace":
        """Unflatten this space from JAX tree operations."""
        return cls(**aux_data)


# Convenience functions
def make_continuous(
    shape: Union[int, Sequence[int]],
    low: ArrayLike = -1.0,
    high: ArrayLike = 1.0,
    dtype: DTypeLike = float,
) -> BoundedArraySpace:
    """Create a continuous (bounded float) space.

    Args:
        shape: Shape of the space.
        low: Lower bounds (inclusive).
        high: Upper bounds (inclusive).
        dtype: Data type for the space.

    Returns:
        A BoundedArraySpace.
    """
    return BoundedArraySpace(shape=shape, minimum=low, maximum=high, dtype=dtype)


def make_discrete(num_values: int, dtype: DTypeLike = int) -> DiscreteSpace:
    """Create a discrete space.

    Args:
        num_values: Number of discrete values.
        dtype: Integer data type for the space.

    Returns:
        A DiscreteSpace.
    """
    return DiscreteSpace(num_values=num_values, dtype=dtype)


# Register spaces as JAX pytrees
jax.tree_util.register_pytree_node(ArraySpace, ArraySpace.tree_flatten, ArraySpace.tree_unflatten)

jax.tree_util.register_pytree_node(
    BoundedArraySpace, BoundedArraySpace.tree_flatten, BoundedArraySpace.tree_unflatten
)

jax.tree_util.register_pytree_node(
    DiscreteSpace, DiscreteSpace.tree_flatten, DiscreteSpace.tree_unflatten
)

jax.tree_util.register_pytree_node(
    MultiDiscreteSpace,
    MultiDiscreteSpace.tree_flatten,
    MultiDiscreteSpace.tree_unflatten,
)

jax.tree_util.register_pytree_node(DictSpace, DictSpace.tree_flatten, DictSpace.tree_unflatten)

jax.tree_util.register_pytree_node(TupleSpace, TupleSpace.tree_flatten, TupleSpace.tree_unflatten)

jax.tree_util.register_pytree_node(
    EnvironmentSpace, EnvironmentSpace.tree_flatten, EnvironmentSpace.tree_unflatten
)
