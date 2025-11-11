"""Replay buffer for offline RL datasets."""

from typing import Iterator
import jax
import jax.numpy as jnp
import numpy as np
from ..core.types import Batch


class ReplayBuffer:
    """Replay buffer for offline RL datasets.
    
    Stores transitions and provides batched sampling for training.
    
    Attributes:
        capacity: Maximum buffer size
        size: Current number of transitions stored
        
    Example:
        buffer = ReplayBuffer(capacity=1_000_000)
        buffer.add(obs, act, reward, next_obs, done)
        batch = buffer.sample(batch_size=256, rng=rng_key)
    """
    
    def __init__(self, capacity: int):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        
        # Storage arrays (initialized lazily)
        self.obs: np.ndarray | None = None
        self.act: np.ndarray | None = None
        self.reward: np.ndarray | None = None
        self.next_obs: np.ndarray | None = None
        self.done: np.ndarray | None = None
    
    def _initialize_storage(self, obs_dim: int, act_dim: int):
        """Initialize storage arrays.
        
        Args:
            obs_dim: Observation dimension
            act_dim: Action dimension
        """
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(self.capacity, dtype=np.float32)
    
    def add(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ):
        """Add a transition to the buffer.
        
        Args:
            obs: Observation
            act: Action
            reward: Reward
            next_obs: Next observation
            done: Done flag
        """
        # Initialize storage on first add
        if self.obs is None:
            obs_dim = obs.shape[0] if obs.ndim == 1 else obs.shape[-1]
            act_dim = act.shape[0] if act.ndim == 1 else act.shape[-1]
            self._initialize_storage(obs_dim, act_dim)
        
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.reward[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, rng: jnp.ndarray) -> Batch:
        """Sample a batch of transitions.
        
        Args:
            batch_size: Batch size
            rng: JAX random key
            
        Returns:
            Batch of transitions
        """
        indices = jax.random.randint(rng, (batch_size,), 0, self.size)
        indices = np.array(indices)
        
        return Batch(
            obs=jnp.array(self.obs[indices]),
            act=jnp.array(self.act[indices]),
            reward=jnp.array(self.reward[indices]),
            next_obs=jnp.array(self.next_obs[indices]),
            done=jnp.array(self.done[indices]),
        )
    
    def load_dataset(self, dataset: dict):
        """Load a complete offline dataset into the buffer.
        
        Args:
            dataset: Dictionary with keys 'observations', 'actions', 'rewards',
                    'next_observations', 'dones'
        """
        obs = dataset["observations"]
        act = dataset["actions"]
        reward = dataset["rewards"]
        next_obs = dataset["next_observations"]
        done = dataset["dones"]
        
        n_transitions = obs.shape[0]
        
        # Initialize storage
        obs_dim = obs.shape[-1]
        act_dim = act.shape[-1]
        self.capacity = max(self.capacity, n_transitions)
        self._initialize_storage(obs_dim, act_dim)
        
        # Load all transitions
        self.obs[:n_transitions] = obs
        self.act[:n_transitions] = act
        self.reward[:n_transitions] = reward
        self.next_obs[:n_transitions] = next_obs
        self.done[:n_transitions] = done
        
        self.size = n_transitions
        self.ptr = 0
    
    def get_iterator(self, batch_size: int, rng: jnp.ndarray) -> Iterator[Batch]:
        """Create an iterator over the buffer.
        
        Args:
            batch_size: Batch size
            rng: JAX random key
            
        Yields:
            Batches of transitions
        """
        while True:
            rng, subrng = jax.random.split(rng)
            yield self.sample(batch_size, subrng)
