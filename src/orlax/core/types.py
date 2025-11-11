"""Type definitions for batches, states, and other typed containers.

Use these types instead of dicts to improve type safety and autocompletion.
"""

from typing import NamedTuple, Any
import jax.numpy as jnp


class Batch(NamedTuple):
    """Training batch containing offline RL transitions.
    
    Attributes:
        obs: Observations [batch_size, obs_dim]
        act: Actions [batch_size, act_dim]
        reward: Rewards [batch_size]
        next_obs: Next observations [batch_size, obs_dim]
        done: Done flags [batch_size]
    """
    obs: jnp.ndarray
    act: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    done: jnp.ndarray


class TrainState(NamedTuple):
    """Training state container for algorithm checkpointing.
    
    Attributes:
        step: Current training step
        params: Model parameters (pytree)
        opt_state: Optimizer state (pytree)
        rng: JAX random key
        extra: Algorithm-specific state (optional)
    """
    step: int
    params: Any  # Pytree of model parameters
    opt_state: Any  # Pytree of optimizer state
    rng: jnp.ndarray
    extra: Any = None  # Algorithm-specific state
