"""Utility functions for seeding, checkpointing, and common operations."""

import os
import pickle
from typing import Any, Optional
import jax
import jax.numpy as jnp
from flax import serialization


def set_seed(seed: int) -> jax.random.PRNGKey:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed
        
    Returns:
        JAX random key
    """
    return jax.random.PRNGKey(seed)


def save_checkpoint(path: str, state: Any, overwrite: bool = True):
    """Save checkpoint using Flax serialization.
    
    Args:
        path: Path to save checkpoint
        state: State object to save (typically TrainState)
        overwrite: Whether to overwrite existing checkpoint
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"Checkpoint already exists at {path}")
    
    # Convert to state dict and save
    state_dict = serialization.to_state_dict(state)
    with open(path, "wb") as f:
        pickle.dump(state_dict, f)


def load_checkpoint(path: str, state: Any) -> Any:
    """Load checkpoint using Flax serialization.
    
    Args:
        path: Path to checkpoint
        state: Template state object (with correct structure)
        
    Returns:
        Restored state object
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    
    with open(path, "rb") as f:
        state_dict = pickle.load(f)
    
    return serialization.from_state_dict(state, state_dict)


def tree_norm(tree: Any) -> float:
    """Compute L2 norm of a pytree.
    
    Args:
        tree: PyTree (e.g., model parameters)
        
    Returns:
        L2 norm as a scalar
    """
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(x ** 2) for x in leaves))


def tree_mean(tree: Any) -> float:
    """Compute mean of all values in a pytree.
    
    Args:
        tree: PyTree
        
    Returns:
        Mean as a scalar
    """
    leaves = jax.tree_util.tree_leaves(tree)
    total = sum(jnp.sum(x) for x in leaves)
    count = sum(x.size for x in leaves)
    return total / count


def get_device(device: Optional[str] = None) -> jax.Device:
    """Get JAX device.
    
    Args:
        device: Device string ('cpu', 'cuda:0', or None for auto)
        
    Returns:
        JAX device
    """
    if device is None:
        # Auto-select: prefer GPU if available
        devices = jax.devices()
        return devices[0]
    elif device == "cpu":
        return jax.devices("cpu")[0]
    elif device.startswith("cuda"):
        # Extract device index if provided
        if ":" in device:
            idx = int(device.split(":")[1])
            return jax.devices("gpu")[idx]
        else:
            return jax.devices("gpu")[0]
    else:
        raise ValueError(f"Unknown device: {device}")


def return_range(dataset: dict, max_episode_steps: int) -> tuple[float, float]:
    """Calculate the min and max returns in a dataset.
    
    Args:
        dataset: Dictionary containing 'rewards' and 'dones' arrays
        max_episode_steps: Maximum steps per episode
        
    Returns:
        Tuple of (min_return, max_return)
    """
    import numpy as np
    
    rewards = dataset['rewards']
    dones = dataset["dones"]
    
    returns = []
    current_return = 0.0
    episode_length = 0
    
    for i in range(len(rewards)):
        current_return += rewards[i]
        episode_length += 1
        
        # Episode ends if done flag is set
        if dones[i] or episode_length >= max_episode_steps:
            returns.append(current_return)
            current_return = 0.0
            episode_length = 0
    
    # Add final episode if it didn't end
    if episode_length > 0:
        returns.append(current_return)
    
    if len(returns) == 0:
        return 0.0, 0.0
    
    return float(np.min(returns)), float(np.max(returns))
