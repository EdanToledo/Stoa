import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from stoa import spaces


class ArraySpaceTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.space = spaces.ArraySpace(shape=(2, 3), dtype=jnp.float32)
        self.rng_key = jax.random.PRNGKey(np.random.randint(0, 1000000))

    @chex.all_variants()
    def test_sample_shape(self) -> None:
        sample = self.variant(self.space.sample)(self.rng_key)
        self.assertEqual(sample.shape, (2, 3))

    @chex.all_variants()
    def test_contains_valid_sample(self) -> None:
        sample = self.variant(self.space.sample)(self.rng_key)
        self.assertTrue(self.space.contains(sample))

    @chex.all_variants()
    def test_dtype(self) -> None:
        sample = self.variant(self.space.sample)(self.rng_key)
        self.assertEqual(sample.dtype, float)


class DiscreteSpaceTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.space = spaces.DiscreteSpace(num_values=5)
        self.rng_key = jax.random.PRNGKey(np.random.randint(0, 1000000))

    @chex.all_variants()
    def test_sample_in_range(self) -> None:
        sample = self.variant(self.space.sample)(self.rng_key)
        self.assertTrue(0 <= sample < 5)

    @chex.all_variants()
    def test_contains(self) -> None:
        self.assertTrue(self.variant(self.space.contains)(3))
        self.assertFalse(self.variant(self.space.contains)(5))


class DictSpaceTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.space = spaces.DictSpace(
            {
                "pos": spaces.ArraySpace((2,), float),
                "id": spaces.DiscreteSpace(10),
            }
        )
        self.rng_key = jax.random.PRNGKey(np.random.randint(0, 1000000))

    @chex.all_variants()
    def test_sample_structure(self) -> None:
        sample = self.variant(self.space.sample)(self.rng_key)
        self.assertIn("pos", sample)
        self.assertIn("id", sample)

    @chex.all_variants()
    def test_contains_sample(self) -> None:
        sample = self.variant(self.space.sample)(self.rng_key)
        self.assertTrue(self.space.contains(sample))


class BoundedArraySpaceTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.space = spaces.BoundedArraySpace(
            shape=(2, 2), dtype=jnp.float32, minimum=0.0, maximum=1.0
        )
        self.rng_key = jax.random.PRNGKey(np.random.randint(0, 1000000))

    @chex.all_variants()
    def test_sample_within_bounds(self) -> None:
        sample = self.variant(self.space.sample)(self.rng_key)
        self.assertTrue(jnp.all(sample >= 0.0) and jnp.all(sample <= 1.0))
        self.assertEqual(sample.shape, (2, 2))
        self.assertEqual(sample.dtype, float)

    def test_contains(self) -> None:
        self.assertTrue(self.space.contains(jnp.array([[0.5, 0.0], [1.0, 0.7]], dtype=jnp.float32)))
        self.assertFalse(
            self.space.contains(jnp.array([[1.1, 0.0], [1.0, 0.7]], dtype=jnp.float32))
        )
        self.assertFalse(
            self.space.contains(jnp.array([[0.5, -0.1], [1.0, 0.7]], dtype=jnp.float32))
        )

    def test_clip(self) -> None:
        arr = jnp.array([[1.5, -0.5], [0.5, 0.7]], dtype=jnp.float32)
        clipped = self.space.clip(arr)
        self.assertTrue(jnp.all(clipped >= 0.0) and jnp.all(clipped <= 1.0))

    def test_zeros_ones(self) -> None:
        self.assertTrue(jnp.all(self.space.zeros() == 0.0))
        self.assertTrue(jnp.all(self.space.ones() == 1.0))

    def test_repr_and_eq(self) -> None:
        s2 = spaces.BoundedArraySpace(shape=(2, 2), dtype=jnp.float32, minimum=0.0, maximum=1.0)
        self.assertEqual(self.space, s2)
        self.assertIsInstance(repr(self.space), str)


class MultiDiscreteSpaceTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.space = spaces.MultiDiscreteSpace(num_values=[3, 4, 2])
        self.rng_key = jax.random.PRNGKey(np.random.randint(0, 1000000))

    @chex.all_variants()
    def test_sample_in_range(self) -> None:
        sample = self.variant(self.space.sample)(self.rng_key)
        self.assertEqual(sample.shape, (3,))
        self.assertTrue(jnp.all(sample >= 0))
        self.assertTrue(jnp.all(sample < jnp.array([3, 4, 2])))

    def test_contains(self) -> None:
        self.assertTrue(self.space.contains(jnp.array([2, 3, 1], dtype=jnp.int32)))
        self.assertFalse(self.space.contains(jnp.array([3, 0, 1], dtype=jnp.int32)))
        self.assertFalse(self.space.contains(jnp.array([1, 2], dtype=jnp.int32)))

    def test_repr_and_eq(self) -> None:
        s2 = spaces.MultiDiscreteSpace(num_values=[3, 4, 2])
        self.assertEqual(self.space, s2)
        self.assertIsInstance(repr(self.space), str)


class TupleSpaceTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.space = spaces.TupleSpace(
            [
                spaces.DiscreteSpace(2),
                spaces.ArraySpace((2,), float),
            ]
        )
        self.rng_key = jax.random.PRNGKey(np.random.randint(0, 1000000))

    @chex.all_variants()
    def test_sample_and_contains(self) -> None:
        sample = self.variant(self.space.sample)(self.rng_key)
        self.assertTrue(self.space.contains(sample))
        self.assertIsInstance(sample, tuple)
        self.assertEqual(len(sample), 2)

    def test_invalid_contains(self) -> None:
        self.assertFalse(self.space.contains((1,)))
        self.assertFalse(self.space.contains((1, jnp.array([0.0, 0.0]), 3)))
        self.assertFalse(self.space.contains([1, jnp.array([0.0, 0.0])]))

    def test_repr_and_eq(self) -> None:
        s2 = spaces.TupleSpace(
            [
                spaces.DiscreteSpace(2),
                spaces.ArraySpace((2,), float),
            ]
        )
        self.assertEqual(self.space, s2)
        self.assertIsInstance(repr(self.space), str)


class UtilityMethodsTest(parameterized.TestCase):
    def test_arrayspace_zeros_ones(self) -> None:
        s = spaces.ArraySpace((2, 2), dtype=jnp.float32)
        self.assertTrue(jnp.all(s.zeros() == 0.0))
        self.assertTrue(jnp.all(s.ones() == 1.0))

    def test_discretespace_one_hot(self) -> None:
        s = spaces.DiscreteSpace(4)
        oh = s.one_hot(jnp.array(2))
        self.assertTrue(jnp.allclose(oh, jnp.array([0.0, 0.0, 1.0, 0.0])))


class PyTreeTest(parameterized.TestCase):
    def test_arrayspace_pytree(self) -> None:
        import jax

        s = spaces.ArraySpace((2, 2), dtype=jnp.float32)
        leaves, treedef = jax.tree_util.tree_flatten(s)
        s2 = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertEqual(s, s2)

    def test_dictspace_pytree(self) -> None:
        import jax

        s = spaces.DictSpace({"a": spaces.DiscreteSpace(2)})
        leaves, treedef = jax.tree_util.tree_flatten(s)
        s2 = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertEqual(s, s2)

    def test_tuplespace_pytree(self) -> None:
        import jax

        s = spaces.TupleSpace([spaces.DiscreteSpace(2), spaces.ArraySpace((1,), float)])
        leaves, treedef = jax.tree_util.tree_flatten(s)
        s2 = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertEqual(s, s2)


class NestedCompositeSpacesTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.space = spaces.DictSpace(
            {
                "tuple": spaces.TupleSpace(
                    [
                        spaces.DiscreteSpace(2),
                        spaces.BoundedArraySpace((2,), minimum=0, maximum=1),
                    ]
                ),
                "array": spaces.ArraySpace((3,), float),
            }
        )

        self.rng_key = jax.random.PRNGKey(np.random.randint(0, 1000000))

    @chex.all_variants()
    def test_sample_and_contains(self) -> None:
        sample = self.variant(self.space.sample)(self.rng_key)
        self.assertTrue(self.space.contains(sample))
        self.assertIn("tuple", sample)
        self.assertIn("array", sample)
        self.assertIsInstance(sample["tuple"], tuple)
        self.assertEqual(len(sample["tuple"]), 2)

    def test_invalid_contains(self) -> None:
        invalid = {
            "tuple": (2, jnp.array([0.5, 0.5])),
            "array": jnp.array([0.0, 0.0, 0.0]),
        }
        self.assertFalse(self.space.contains(invalid))


class SpaceEdgeCaseTest(parameterized.TestCase):
    def test_bounded_array_space_min_eq_max(self) -> None:
        s = spaces.BoundedArraySpace(shape=(2, 2), minimum=1.0, maximum=1.0)
        sample = s.sample(jax.random.PRNGKey(0))
        self.assertTrue(jnp.all(sample == 1.0))
        self.assertTrue(s.contains(sample))

    def test_bounded_array_space_nonfinite_bounds(self) -> None:
        s = spaces.BoundedArraySpace(shape=(2, 2), minimum=-jnp.inf, maximum=jnp.inf)
        sample = s.sample(jax.random.PRNGKey(0))
        self.assertEqual(sample.shape, (2, 2))

    def test_bounded_array_space_broadcasting(self) -> None:
        s = spaces.BoundedArraySpace(shape=(2, 2), minimum=0.0, maximum=[1.0, 2.0])
        sample = s.sample(jax.random.PRNGKey(0))
        self.assertEqual(sample.shape, (2, 2))
        self.assertTrue(jnp.all(sample[:, 0] <= 1.0))
        self.assertTrue(jnp.all(sample[:, 1] <= 2.0))


class SpaceInvalidConstructionTest(parameterized.TestCase):
    def test_discrete_space_invalid_num_values(self) -> None:
        with self.assertRaises(ValueError):
            spaces.DiscreteSpace(num_values=0)
        with self.assertRaises(ValueError):
            spaces.DiscreteSpace(num_values=-1)

    def test_discrete_space_invalid_dtype(self) -> None:
        with self.assertRaises(ValueError):
            spaces.DiscreteSpace(num_values=5, dtype=jnp.float32)

    def test_multidiscrete_space_invalid_num_values(self) -> None:
        with self.assertRaises(ValueError):
            spaces.MultiDiscreteSpace(num_values=[3, 0, 2])
        with self.assertRaises(ValueError):
            spaces.MultiDiscreteSpace(num_values=[-1, 2, 2])

    def test_multidiscrete_space_invalid_dtype(self) -> None:
        with self.assertRaises(ValueError):
            spaces.MultiDiscreteSpace(num_values=[3, 4], dtype=jnp.float32)

    def test_bounded_array_space_invalid_bounds(self) -> None:
        with self.assertRaises(ValueError):
            spaces.BoundedArraySpace(shape=(2, 2), minimum=2.0, maximum=1.0)


class SpaceDeterminismTest(parameterized.TestCase):
    def test_array_space_sample_determinism(self) -> None:
        s = spaces.ArraySpace((2, 2), dtype=jnp.float32)
        key = jax.random.PRNGKey(42)
        sample1 = s.sample(key)
        sample2 = s.sample(key)
        self.assertTrue(jnp.all(sample1 == sample2))

    def test_discrete_space_sample_determinism(self) -> None:
        s = spaces.DiscreteSpace(5)
        key = jax.random.PRNGKey(42)
        sample1 = s.sample(key)
        sample2 = s.sample(key)
        self.assertTrue(jnp.all(sample1 == sample2))

    def test_multidiscrete_space_sample_determinism(self) -> None:
        s = spaces.MultiDiscreteSpace([3, 4, 2])
        key = jax.random.PRNGKey(42)
        sample1 = s.sample(key)
        sample2 = s.sample(key)
        self.assertTrue(jnp.all(sample1 == sample2))


class EnvironmentSpaceTest(parameterized.TestCase):
    def setUp(self) -> None:
        self.env_space = spaces.EnvironmentSpace(
            observations=spaces.ArraySpace((2,), dtype=jnp.float32),
            actions=spaces.DiscreteSpace(3),
            rewards=spaces.BoundedArraySpace((), minimum=-1.0, maximum=1.0),
            discounts=spaces.BoundedArraySpace((), minimum=0.0, maximum=1.0),
            state=spaces.ArraySpace((4,), dtype=jnp.float32),
        )
        self.rng_key = jax.random.PRNGKey(0)

    def test_sample_timestep_and_contains(self) -> None:
        timestep = self.env_space.sample_timestep(self.rng_key)
        self.assertTrue(self.env_space.contains(timestep))

    def test_sample_transition(self) -> None:
        transition = self.env_space.sample_transition(self.rng_key)
        self.assertIn("observation", transition)
        self.assertIn("action", transition)
        self.assertIn("reward", transition)
        self.assertIn("discount", transition)

    def test_sample_state(self) -> None:
        state = self.env_space.sample_state(self.rng_key)
        self.assertEqual(state.shape, (4,))


if __name__ == "__main__":
    absltest.main()
