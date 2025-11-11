"""High-level experiment runner for training offline RL algorithms."""

from typing import Dict
from collections import deque
import jax
import jax.numpy as jnp
import numpy as np

from ..core.cfg import TrainingCfg, ModelCfg, AlgoCfg, EvalCfg
from ..core.logger import Logger
from ..core.utils import set_seed, save_checkpoint
from ..data.minari import MinariDataset
from ..data.dataset import Dataset
from ..algos import ALGORITHMS


class ExperimentRunner:
    """High-level runner for offline RL experiments.
    
    Manages the full training loop including:
    - Dataset loading
    - Algorithm initialization
    - Training with logging
    - Periodic evaluation
    - Checkpointing
    
    Example:
        runner = ExperimentRunner(cfg, model_cfg, algo_cfg, eval_cfg)
        runner.run()
    """
    
    def __init__(
        self,
        cfg: TrainingCfg,
        model_cfg: ModelCfg,
        algo_cfg: AlgoCfg,
        eval_cfg: EvalCfg,
    ):
        """Initialize experiment runner.
        
        Args:
            cfg: Training configuration
            model_cfg: Model architecture configuration
            algo_cfg: Algorithm-specific configuration
            eval_cfg: Evaluation configuration
        """
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.algo_cfg = algo_cfg
        self.eval_cfg = eval_cfg
        
        # Initialize logger with group and run name
        group_name = f"{cfg.algo}_{cfg.env_name}"
        run_name = f"{cfg.algo}_{cfg.env_name}_{cfg.seed}"
        
        self.logger = Logger(
            config=cfg,
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=run_name,
            group=group_name,
            mode=cfg.wandb_mode,
        )
        
        # Set random seed
        self.rng = set_seed(cfg.seed)
        
        # Load dataset
        self.dataset, self.eval_env = self._load_dataset()
        
        # Initialize algorithm
        self.algorithm = self._init_algorithm()
        self.state = None
        
        # Evaluation tracking
        self.eval_rewards = deque(maxlen=eval_cfg.moving_avg_window)
    
    def _load_dataset(self):
        """Load offline dataset."""
        print(f"Loading dataset: {self.cfg.env_name}")
        
        minari_dataset = MinariDataset(self.cfg.env_name)
        raw_data = minari_dataset.get_dataset()
        eval_env = minari_dataset.get_env()
        
        # Wrap in Dataset class for normalization
        dataset = Dataset(
            data=raw_data,
            normalize_obs=True,
            normalize_act=False,
            normalize_rewards=True,
            max_episode_steps=eval_env.spec.max_episode_steps,
        )
        
        # Store normalization statistics for evaluation
        self.obs_mean = dataset.obs_mean
        self.obs_std = dataset.obs_std
        
        stats = dataset.get_statistics()
        print(f"Dataset stats: {stats}")
        
        return dataset, eval_env
    
    def _init_algorithm(self):
        """Initialize the offline RL algorithm."""
        algo_class = ALGORITHMS.get(self.cfg.algo)
        if algo_class is None:
            raise ValueError(
                f"Unknown algorithm: {self.cfg.algo}. "
                f"Available: {list(ALGORITHMS.keys())}"
            )
        
        obs_dim = self.dataset.data["observations"].shape[-1]
        act_dim = self.dataset.data["actions"].shape[-1]
        
        return algo_class(
            obs_dim=obs_dim,
            act_dim=act_dim,
            cfg=self.cfg,
            model_cfg=self.model_cfg,
            algo_cfg=self.algo_cfg,
        )
    
    def run(self):
        """Run the full training loop."""
        print(f"Starting training: {self.cfg.algo} on {self.cfg.env_name}")
        
        # Initialize algorithm state
        self.rng, init_rng = jax.random.split(self.rng)
        self.state = self.algorithm.init(init_rng)
        
        # Get replay buffer
        buffer = self.dataset.get_buffer()
        
        # Initialize progress bar
        self.logger.init_progress(self.cfg.total_steps, desc="Training")
        
        # Training loop
        for step in range(self.cfg.total_steps):
            # Sample batch
            self.rng, batch_rng = jax.random.split(self.rng)
            batch = buffer.sample(self.cfg.batch_size, batch_rng)
            
            # Update
            self.rng, update_rng = jax.random.split(self.rng)
            self.state, train_metrics = self.algorithm.update(
                self.state, batch, update_rng
            )
            
            # Evaluate if needed
            should_eval = (step % self.cfg.eval_freq == 0 and step > 0)
            if should_eval:
                eval_metrics = self._evaluate()
                # Merge eval metrics with training metrics
                train_metrics = {**train_metrics, **eval_metrics}
                
                # Update progress bar with eval info
                if len(self.eval_rewards) > 0:
                    ma = np.mean(self.eval_rewards)
                    self.logger.update_progress(
                        {"eval_reward_ma": f"{ma:.2f}"}, n=0
                    )
            
            # Checkpoint
            if step % self.cfg.checkpoint_freq == 0 and step > 0:
                checkpoint_path = f"checkpoints/{self.cfg.algo}_{self.cfg.env_name}_step_{step}.pkl"
                save_checkpoint(checkpoint_path, self.state)
                print(f"\nCheckpoint saved: {checkpoint_path}")
            
            # Log metrics (combined if evaluation happened)
            if step % self.cfg.log_freq == 0:
                self.logger.log(train_metrics, step=step)

            self.logger.update_progress() # Update progress bar
        
        # Final evaluation
        print("\nFinal evaluation...")
        final_metrics = self._evaluate(num_episodes=self.eval_cfg.num_episodes * 2)
        # Use a step beyond the training loop to avoid conflicts
        self.logger.log(final_metrics, step=self.cfg.total_steps)
        
        # Cleanup
        self.logger.close_progress()
        self.logger.finish()
        
        print("Training complete!")
    
    def _evaluate(self, num_episodes: int | None = None) -> Dict[str, float]:
        """Run evaluation episodes.
        
        Args:
            num_episodes: Number of episodes to run (uses eval_cfg default if None)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if num_episodes is None:
            num_episodes = self.eval_cfg.num_episodes
        
        episode_rewards = []
        
        for episode_idx in range(num_episodes):
            obs, info = self.eval_env.reset()
            terminated = False
            truncated = False
            episode_reward = 0.0
            step_count = 0
            
            while not (terminated or truncated):
                # Normalize observation for policy (matching training distribution)
                if self.obs_mean is not None and self.obs_std is not None:
                    obs_normalized = (obs - np.array(self.obs_mean)) / np.array(self.obs_std)
                else:
                    obs_normalized = obs
                
                # Check for NaN/Inf in observations
                if not np.all(np.isfinite(obs_normalized)):
                    print(f"Warning: Invalid observation at step {step_count}, skipping episode")
                    break
                
                # Get action from policy (standard interface for all algorithms)
                obs_tensor = jnp.array(obs_normalized[None, :])
                action = self.algorithm.eval_step(self.state, obs_tensor)
                action = np.array(action[0])
                
                # Check for NaN/Inf in actions
                if not np.all(np.isfinite(action)):
                    print(f"Warning: Invalid action at step {step_count}, skipping episode")
                    break
                
                action = np.clip(
                    action,
                    self.eval_env.action_space.low,
                    self.eval_env.action_space.high
                )
                
                # Step environment
                try:
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    episode_reward += reward
                    step_count += 1
                except Exception as e:
                    print(f"Warning: Environment step failed at step {step_count}: {e}")
                    break
            
            episode_rewards.append(episode_reward)
        
        # Compute statistics
        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        
        # Update moving average
        self.eval_rewards.append(mean_reward)
        moving_avg = float(np.mean(self.eval_rewards))
        
        return {
            "eval/total_reward": mean_reward,
            "eval/total_reward_std": std_reward,
            "eval/total_reward_ma": moving_avg,
        }
