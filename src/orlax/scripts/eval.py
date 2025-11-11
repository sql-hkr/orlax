"""CLI script for evaluating trained models."""

import argparse
import jax
import jax.numpy as jnp
import numpy as np

from orlax.core.utils import load_checkpoint, set_seed
from orlax.data.minari import MinariDataset
from orlax.algos import ALGORITHMS


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate offline RL models")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Environment name"
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        help="Algorithm name"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render episodes"
    )
    
    args = parser.parse_args()
    
    # Set seed
    rng = set_seed(args.seed)
    
    # Load environment
    print(f"Loading environment: {args.env}")
    minari_dataset = MinariDataset(args.env)
    env = minari_dataset.get_env()
    
    # Get normalization statistics
    from orlax.data.dataset import Dataset
    raw_data = minari_dataset.get_dataset()
    dataset = Dataset(
        data=raw_data,
        normalize_obs=True,
        normalize_act=False,
    )
    obs_mean = dataset.obs_mean
    obs_std = dataset.obs_std
    print("Loaded normalization statistics from dataset")
    
    # Load algorithm (just for structure)
    algo_class = ALGORITHMS[args.algo]
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    # Create dummy config for initialization
    from orlax.core.cfg import TrainingCfg, ModelCfg, AlgoCfg
    dummy_cfg = TrainingCfg(algo=args.algo, env_name=args.env)
    dummy_model_cfg = ModelCfg()
    dummy_algo_cfg = AlgoCfg()
    
    algorithm = algo_class(
        obs_dim=obs_dim,
        act_dim=act_dim,
        cfg=dummy_cfg,
        model_cfg=dummy_model_cfg,
        algo_cfg=dummy_algo_cfg,
    )
    
    # Initialize and load checkpoint
    rng, init_rng = jax.random.split(rng)
    state = algorithm.init(init_rng)
    state = load_checkpoint(args.checkpoint, state)
    
    print(f"Loaded checkpoint: {args.checkpoint}")
    
    # Evaluate
    episode_rewards = []
    
    for ep in range(args.num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        
        while not done:
            obs_tensor = jnp.array(obs[None, :])
            
            # Normalize observation (matching training distribution)
            if obs_mean is not None and obs_std is not None:
                obs_tensor = (obs_tensor - obs_mean) / obs_std
            
            action = algorithm.eval_step(state, obs_tensor)
            action = np.array(action[0])
            
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1
            
            if args.render:
                env.render()
        
        episode_rewards.append(episode_reward)
        print(f"Episode {ep + 1}: reward = {episode_reward:.2f}, steps = {steps}")
    
    # Print statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print("\n" + "=" * 80)
    print(f"Evaluation Results ({args.num_episodes} episodes)")
    print("=" * 80)
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
