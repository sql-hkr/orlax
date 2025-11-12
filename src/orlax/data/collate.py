"""Batching and sharding utilities for multi-device training."""

import jax
import jax.numpy as jnp
from typing import Any
from ..core.types import Batch


def collate_batch(samples: list[Batch]) -> Batch:
    """Collate a list of samples into a single batch.
    
    Args:
        samples: List of Batch objects
        
    Returns:
        Single Batch with stacked arrays
    """
    return Batch(
        obs=jnp.stack([s.obs for s in samples]),
        act=jnp.stack([s.act for s in samples]),
        reward=jnp.stack([s.reward for s in samples]),
        next_obs=jnp.stack([s.next_obs for s in samples]),
        done=jnp.stack([s.done for s in samples]),
    )


def shard_batch(batch: Batch, n_devices: int) -> Batch:
    """Shard a batch across multiple devices for pmap.
    
    Args:
        batch: Batch to shard
        n_devices: Number of devices
        
    Returns:
        Sharded batch with leading device axis
        
    Example:
        batch = Batch(...)  # shape: [256, ...]
        sharded = shard_batch(batch, n_devices=4)  # shape: [4, 64, ...]
    """
    # Reshape to add device dimension
    def shard_array(x: jnp.ndarray) -> jnp.ndarray:
        batch_size = x.shape[0]
        assert batch_size % n_devices == 0, (
            f"Batch size {batch_size} must be divisible by n_devices {n_devices}"
        )
        per_device = batch_size // n_devices
        return x.reshape(n_devices, per_device, *x.shape[1:])
    
    return jax.tree.map(shard_array, batch)


def unshard_batch(batch: Batch) -> Batch:
    """Unshard a batch by merging the device dimension.
    
    Args:
        batch: Sharded batch with leading device axis
        
    Returns:
        Unsharded batch
        
    Example:
        sharded = Batch(...)  # shape: [4, 64, ...]
        batch = unshard_batch(sharded)  # shape: [256, ...]
    """
    def unshard_array(x: jnp.ndarray) -> jnp.ndarray:
        return x.reshape(-1, *x.shape[2:])
    
    return jax.tree.map(unshard_array, batch)


def prepare_batch_for_device(batch: Batch, device: jax.Device) -> Batch:
    """Move a batch to a specific device.
    
    Args:
        batch: Batch to move
        device: Target device
        
    Returns:
        Batch on the target device
    """
    return jax.tree.map(lambda x: jax.device_put(x, device), batch)
