"""Dataset class with normalization and caching utilities."""

from typing import Optional
import numpy as np
from ..core.buffer import ReplayBuffer


class Dataset:
    """Dataset wrapper with normalization and preprocessing.
    
    Provides utilities for normalizing observations/actions and creating
    batched iterators for training.
    
    Example:
        dataset = Dataset(data_dict)
        dataset.normalize_observations()
        batch_iter = dataset.get_iterator(batch_size=256, rng=rng_key)
    """
    
    def __init__(self, data: dict, normalize_obs: bool = False, normalize_act: bool = False, normalize_rewards: bool = False, max_episode_steps: Optional[int] = None):
        """Initialize dataset.
        
        Args:
            data: Dictionary with keys 'observations', 'actions', 'rewards',
                 'next_observations', 'dones'
            normalize_obs: Whether to normalize observations
            normalize_act: Whether to normalize actions
            normalize_rewards: Whether to normalize rewards
            max_episode_steps: Maximum steps per episode (needed for reward normalization)
        """
        self.data = data
        self.size = data["observations"].shape[0]
        
        # Compute normalization statistics
        self.obs_mean: Optional[np.ndarray] = None
        self.obs_std: Optional[np.ndarray] = None
        self.act_mean: Optional[np.ndarray] = None
        self.act_std: Optional[np.ndarray] = None
        self.reward_scale: Optional[float] = None
        
        if normalize_obs:
            self.normalize_observations()
        if normalize_act:
            self.normalize_actions()
        if normalize_rewards:
            if max_episode_steps is None:
                raise ValueError("max_episode_steps must be provided for reward normalization")
            self.normalize_rewards(max_episode_steps)
    
    def normalize_observations(self):
        """Normalize observations to zero mean and unit variance."""
        obs = self.data["observations"]
        self.obs_mean = np.mean(obs, axis=0)
        self.obs_std = np.std(obs, axis=0) + 1e-8
        
        self.data["observations"] = (obs - self.obs_mean) / self.obs_std
        self.data["next_observations"] = (
            self.data["next_observations"] - self.obs_mean
        ) / self.obs_std
    
    def normalize_actions(self):
        """Normalize actions to zero mean and unit variance."""
        actions = self.data["actions"]
        self.act_mean = np.mean(actions, axis=0)
        self.act_std = np.std(actions, axis=0) + 1e-8
        
        self.data["actions"] = (actions - self.act_mean) / self.act_std
    
    def normalize_rewards(self, max_episode_steps: int):
        """Normalize rewards using return range.
        
        Divides rewards by the return range and scales by max_episode_steps,
        following the approach used in D4RL benchmarks for MuJoCo tasks.
        
        Args:
            max_episode_steps: Maximum steps per episode
        """
        from ..core.utils import return_range
        
        min_ret, max_ret = return_range(self.data, max_episode_steps)
        print(f"Dataset returns have range [{min_ret:.2f}, {max_ret:.2f}]")
        
        if max_ret - min_ret > 0:
            self.reward_scale = max_ret - min_ret
            self.data["rewards"] = self.data["rewards"] / (max_ret - min_ret)
            self.data["rewards"] = self.data["rewards"] * max_episode_steps
            print(f"Rewards normalized: scaled by {max_episode_steps / (max_ret - min_ret):.4f}")
        else:
            print("Warning: Return range is zero, skipping reward normalization")
    
    def denormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Denormalize actions back to original scale.
        
        Args:
            actions: Normalized actions
            
        Returns:
            Denormalized actions
        """
        if self.act_mean is None or self.act_std is None:
            return actions
        return actions * self.act_std + self.act_mean
    
    def get_buffer(self) -> ReplayBuffer:
        """Create a replay buffer from the dataset.
        
        Returns:
            ReplayBuffer containing all transitions
        """
        buffer = ReplayBuffer(capacity=self.size)
        buffer.load_dataset(self.data)
        return buffer
    
    def get_statistics(self) -> dict:
        """Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        return {
            "size": self.size,
            "obs_dim": self.data["observations"].shape[-1],
            "act_dim": self.data["actions"].shape[-1],
            "mean_reward": float(np.mean(self.data["rewards"])),
            "std_reward": float(np.std(self.data["rewards"])),
            "mean_episode_length": self._compute_mean_episode_length(),
        }
    
    def _compute_mean_episode_length(self) -> float:
        """Compute mean episode length."""
        dones = self.data.get("dones", self.data.get("terminals", np.zeros(self.size)))
        n_episodes = np.sum(dones)
        if n_episodes == 0:
            return float(self.size)
        return float(self.size / n_episodes)


def normalize_dataset(dataset: dict) -> tuple[dict, dict]:
    """Normalize a dataset and return statistics.
    
    Args:
        dataset: Dictionary with transition data
        
    Returns:
        Tuple of (normalized_dataset, normalization_stats)
    """
    obs = dataset["observations"]
    actions = dataset["actions"]
    
    # Compute stats
    obs_mean = np.mean(obs, axis=0)
    obs_std = np.std(obs, axis=0) + 1e-8
    act_mean = np.mean(actions, axis=0)
    act_std = np.std(actions, axis=0) + 1e-8
    
    # Normalize
    normalized = {
        "observations": (obs - obs_mean) / obs_std,
        "actions": (actions - act_mean) / act_std,
        "rewards": dataset["rewards"],
        "next_observations": (dataset["next_observations"] - obs_mean) / obs_std,
        "dones": dataset["dones"],
    }
    
    stats = {
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "act_mean": act_mean,
        "act_std": act_std,
    }
    
    return normalized, stats
