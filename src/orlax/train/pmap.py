"""Multi-device training utilities using pmap."""

import jax
import jax.numpy as jnp
from typing import Any


def setup_pmap(n_devices: int | None = None) -> tuple[int, list[jax.Device]]:
    """Setup for multi-device training with pmap.
    
    Args:
        n_devices: Number of devices to use (None for all available)
        
    Returns:
        Tuple of (n_devices, devices)
    """
    devices = jax.devices()
    
    if n_devices is None:
        n_devices = len(devices)
    else:
        devices = devices[:n_devices]
    
    print(f"Using {n_devices} devices: {devices}")
    
    return n_devices, devices


def shard_state(state: Any, n_devices: int) -> Any:
    """Shard state across multiple devices for pmap.
    
    Args:
        state: State to shard
        n_devices: Number of devices
        
    Returns:
        Sharded state with leading device dimension
    """
    def shard_array(x):
        if isinstance(x, jnp.ndarray):
            # Replicate across devices
            return jnp.stack([x] * n_devices)
        return x
    
    return jax.tree.map(shard_array, state)


def unshard_state(state: Any) -> Any:
    """Remove device dimension from sharded state.
    
    Args:
        state: Sharded state
        
    Returns:
        State from first device
    """
    def unshard_array(x):
        if isinstance(x, jnp.ndarray) and x.ndim > 0:
            return x[0]
        return x
    
    return jax.tree.map(unshard_array, state)


def replicate_state(state: Any, n_devices: int) -> Any:
    """Replicate state across devices (for data-parallel training).
    
    Args:
        state: State to replicate
        n_devices: Number of devices
        
    Returns:
        Replicated state
    """
    return jax.device_put_replicated(state, jax.devices()[:n_devices])
