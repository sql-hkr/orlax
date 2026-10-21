"""Training loop utilities."""

import jax
import jax.numpy as jnp
from typing import Any, Dict, Callable

from ..core.types import Batch


def train_step(
    state: Dict[str, Any],
    batch: Batch,
    rng: jnp.ndarray,
    update_fn: Callable,
) -> tuple[Dict[str, Any], Dict[str, float]]:
    """Perform a single training step.
    
    Args:
        state: Algorithm state
        batch: Training batch
        rng: JAX random key
        update_fn: Algorithm update function
        
    Returns:
        Tuple of (new_state, metrics)
    """
    return update_fn(state, batch, rng)


def eval_episode(
    state: Dict[str, Any],
    env: Any,
    eval_fn: Callable,
    max_steps: int = 1000,
) -> float:
    """Evaluate policy for one episode.
    
    Args:
        state: Algorithm state
        env: Gym environment
        eval_fn: Function to get actions from state
        max_steps: Maximum episode length
        
    Returns:
        Total episode reward
    """
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    done = False
    total_reward = 0.0
    steps = 0
    
    while not done and steps < max_steps:
        obs_tensor = jnp.array(obs[None, :])
        action = eval_fn(state, obs_tensor)
        action = jax.device_get(action[0])
        
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, _ = step_out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, _ = step_out
        total_reward += float(reward)
        steps += 1
    
    return total_reward
