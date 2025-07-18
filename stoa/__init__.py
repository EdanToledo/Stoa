"""Stoa: A JAX-Native Interface for Reinforcement Learning Environments."""

# Version
__version__ = "0.1.0"

# Core wrappers
from stoa.core_wrappers.auto_reset import AutoResetWrapper
from stoa.core_wrappers.episode_metrics import (
    RecordEpisodeMetrics,
    get_final_step_metrics,
)
from stoa.core_wrappers.wrapper import AddRNGKey, StateWithKey, Wrapper, WrapperState

# Core types and abstractions
from stoa.env_types import (
    Action,
    ActionMask,
    Discount,
    EnvParams,
    Observation,
    Reward,
    State,
    StepCount,
    StepType,
    TimeStep,
    TimeStepExtras,
)

# Environment adapters
from stoa.env_wrappers.brax import BraxToStoa
from stoa.env_wrappers.gymnax import GymnaxToStoa
from stoa.env_wrappers.jumanji import JumanjiToStoa
from stoa.env_wrappers.kinetix import KinetixToStoa
from stoa.env_wrappers.navix import NavixToStoa
from stoa.env_wrappers.pgx import PGXToStoa
from stoa.env_wrappers.playground import MuJoCoPlaygroundToStoa
from stoa.env_wrappers.xminigrid import XMiniGridToStoa
from stoa.environment import Environment

# Spaces
from stoa.spaces import (
    ArraySpace,
    BoundedArraySpace,
    DictSpace,
    DiscreteSpace,
    EnvironmentSpace,
    MultiDiscreteSpace,
    Space,
    TupleSpace,
    make_continuous,
    make_discrete,
)

# Action space wrappers
from stoa.utility_wrappers.action_space_transforms import (
    MultiBoundedToBoundedWrapper,
    MultiDiscreteToDiscreteWrapper,
)
from stoa.utility_wrappers.consistent_extras import ConsistentExtrasWrapper
from stoa.utility_wrappers.flatten_obs import FlattenObservationWrapper
from stoa.utility_wrappers.frame_stacking import FrameStackingWrapper

# Observation wrappers
from stoa.utility_wrappers.obs_extract import ObservationExtractWrapper
from stoa.utility_wrappers.obs_transforms import (
    AddActionMaskWrapper,
    AddStartFlagAndPrevAction,
    AddStepCountWrapper,
    MakeChannelLast,
    ObservationTypeWrapper,
)
from stoa.utility_wrappers.step_limit import EpisodeStepLimitWrapper

__all__ = [
    # Version
    "__version__",
    # Core types
    "Action",
    "ActionMask",
    "Discount",
    "EnvParams",
    "Observation",
    "Reward",
    "State",
    "StepCount",
    "StepType",
    "TimeStep",
    "TimeStepExtras",
    "Environment",
    # Spaces
    "ArraySpace",
    "BoundedArraySpace",
    "DictSpace",
    "DiscreteSpace",
    "EnvironmentSpace",
    "MultiDiscreteSpace",
    "Space",
    "TupleSpace",
    "make_continuous",
    "make_discrete",
    # Core wrappers
    "Wrapper",
    "WrapperState",
    "AddRNGKey",
    "StateWithKey",
    "AutoResetWrapper",
    "RecordEpisodeMetrics",
    "get_final_step_metrics",
    # Observation wrappers
    "ObservationExtractWrapper",
    "AddStartFlagAndPrevAction",
    "MakeChannelLast",
    "FlattenObservationWrapper",
    "FrameStackingWrapper",
    "AddStepCountWrapper",
    "AddActionMaskWrapper",
    "ObservationTypeWrapper",
    # Utility wrappers
    "EpisodeStepLimitWrapper",
    "ConsistentExtrasWrapper",
    # Action wrappers
    "MultiDiscreteToDiscreteWrapper",
    "MultiBoundedToBoundedWrapper",
    # Environment adapters
    "BraxToStoa",
    "GymnaxToStoa",
    "JumanjiToStoa",
    "NavixToStoa",
    "XMiniGridToStoa",
    "KinetixToStoa",
    "MuJoCoPlaygroundToStoa",
    "PGXToStoa",
]
