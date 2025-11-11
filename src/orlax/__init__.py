"""ORLax - Offline Reinforcement Learning with JAX.

An extensible, research-friendly offline RL framework built with JAX, Flax, and Optax.
"""

__version__ = "0.1.0"

# Core utilities
from .core.cfg import TrainingCfg, ModelCfg, AlgoCfg, EvalCfg
from .core.types import Batch, TrainState
from .core.logger import Logger
from .core.utils import set_seed, save_checkpoint, load_checkpoint, return_range
from .core.optim import create_optimizer, create_lr_schedule
from .core.buffer import ReplayBuffer

# Data loading
from .data.minari import MinariDataset
from .data.dataset import Dataset, normalize_dataset

# Models
from .models.mlp import MLP
from .models.nets import Actor, Critic, ValueNetwork

# Algorithms
from .algos import BC, CQL, IQL, ALGORITHMS

# Training
from .train.runner import ExperimentRunner

__all__ = [
    # Version
    "__version__",
    # Core
    "TrainingCfg",
    "ModelCfg",
    "AlgoCfg",
    "EvalCfg",
    "Batch",
    "TrainState",
    "Logger",
    "set_seed",
    "save_checkpoint",
    "load_checkpoint",
    "return_range",
    "create_optimizer",
    "create_lr_schedule",
    "ReplayBuffer",
    # Data
    "MinariDataset",
    "Dataset",
    "normalize_dataset",
    # Models
    "MLP",
    "Actor",
    "Critic",
    "ValueNetwork",
    "QNetwork",
    # Algorithms
    "BC",
    "CQL",
    "IQL",
    "ALGORITHMS",
    # Training
    "ExperimentRunner",
]

