"""Minari dataset adapter for loading offline RL datasets."""

from typing import Optional, Any
import numpy as np

try:
    import minari
    import gymnasium as gym
    MINARI_AVAILABLE = True
    GymEnv = gym.Env
except ImportError:
    MINARI_AVAILABLE = False
    GymEnv = Any  # Fallback type when minari not available
    print("Warning: minari not available. Install with: pip install 'minari[all]'")


class MinariDataset:
    """Adapter for Minari datasets.
    
    Loads Minari datasets and converts them to the project's standardized format.
    Minari is the successor to D4RL with better maintenance and Gymnasium support.
    
    Note: Requires minari to be installed:
        pip install 'minari[all]'
    
    Attributes:
        dataset_id: Minari dataset ID
        dataset: Minari dataset object
        
    Example:
        dataset = MinariDataset("D4RL/door/human-v2")
        data = dataset.get_dataset()
        obs, act, reward, next_obs, done = data['observations'], ...
    """
    
    def __init__(self, dataset_id: str, download: bool = True):
        """Initialize Minari dataset.
        
        Args:
            dataset_id: Minari dataset ID (e.g., 'D4RL/door/human-v2')
            download: Whether to automatically download the dataset if not found locally
        """
        if not MINARI_AVAILABLE:
            raise ImportError(
                "minari is not installed. Install with: "
                "pip install 'minari[all]'"
            )
        
        self.dataset_id = dataset_id
        
        # Try to load dataset, download if not found
        try:
            self.dataset = minari.load_dataset(dataset_id)
        except FileNotFoundError:
            if download:
                print(f"Dataset {dataset_id} not found locally. Downloading...")
                minari.download_dataset(dataset_id)
                self.dataset = minari.load_dataset(dataset_id)
                print(f"Dataset {dataset_id} downloaded successfully.")
            else:
                raise
        
        self.env = None
        
        # Preprocess dataset
        self._preprocess()
    
    def _preprocess(self):
        """Preprocess the dataset (extract transitions from episodes)."""
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        
        # Iterate through all episodes
        for episode in self.dataset:
            obs = episode.observations
            acts = episode.actions
            rews = episode.rewards
            terms = episode.terminations
            truncs = episode.truncations
            
            # Handle different observation shapes (dict vs array)
            if isinstance(obs, dict):
                # For dict observations, concatenate all values
                obs = np.concatenate([v.reshape(len(v), -1) for v in obs.values()], axis=-1)
            
            # Get episode length
            ep_len = len(rews)
            
            # Add transitions
            observations.append(obs[:-1])  # All but last
            actions.append(acts)
            rewards.append(rews)
            next_observations.append(obs[1:])  # All but first
            
            # Create done flags (terminated OR truncated)
            done_flags = np.zeros(ep_len, dtype=np.float32)
            if terms[-1] or truncs[-1]:  # Episode ended (terminated or truncated)
                done_flags[-1] = 1.0
            dones.append(done_flags)
        
        # Concatenate all episodes
        self.data = {
            "observations": np.concatenate(observations, axis=0).astype(np.float32),
            "actions": np.concatenate(actions, axis=0).astype(np.float32),
            "rewards": np.concatenate(rewards, axis=0).astype(np.float32),
            "next_observations": np.concatenate(next_observations, axis=0).astype(np.float32),
            "dones": np.concatenate(dones, axis=0).astype(np.float32),
        }
    
    def get_dataset(self) -> dict:
        """Get the preprocessed dataset.
        
        Returns:
            Dictionary with keys: observations, actions, rewards,
            next_observations, dones
        """
        return self.data
    
    def get_env(self) -> GymEnv:
        """Get the gymnasium environment for evaluation.
        
        Returns:
            Gymnasium environment for evaluation
        """
        if self.env is None:
            self.env = self.dataset.recover_environment()
        return self.env
    
    def get_normalizer_stats(self) -> dict:
        """Compute normalization statistics from the dataset.
        
        Returns:
            Dictionary with 'obs_mean', 'obs_std', 'act_mean', 'act_std'
        """
        obs = self.data["observations"]
        actions = self.data["actions"]
        
        return {
            "obs_mean": np.mean(obs, axis=0),
            "obs_std": np.std(obs, axis=0) + 1e-8,
            "act_mean": np.mean(actions, axis=0),
            "act_std": np.std(actions, axis=0) + 1e-8,
        }
    
    @staticmethod
    def list_available_datasets() -> list[str]:
        """List all available Minari datasets.
        
        Returns:
            List of dataset IDs
        """
        if not MINARI_AVAILABLE:
            return []
        
        return minari.list_remote_datasets()
    
    @staticmethod
    def download_dataset(dataset_id: str):
        """Download a Minari dataset if not already cached.
        
        Args:
            dataset_id: Minari dataset ID to download
        """
        if not MINARI_AVAILABLE:
            raise ImportError(
                "minari is not installed. Install with: "
                "pip install 'minari[all]'"
            )
        
        minari.download_dataset(dataset_id)
