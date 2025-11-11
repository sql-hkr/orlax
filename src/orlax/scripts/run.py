"""CLI script for running offline RL experiments."""

import argparse
import tomllib
from dataclasses import asdict

from orlax.core.cfg import TrainingCfg, ModelCfg, AlgoCfg, EvalCfg
from orlax.train.runner import ExperimentRunner


def load_config_from_toml(path: str) -> tuple[TrainingCfg, ModelCfg, AlgoCfg, EvalCfg]:
    """Load configuration from TOML file.
    
    Args:
        path: Path to TOML config file
        
    Returns:
        Tuple of (training_cfg, model_cfg, algo_cfg, eval_cfg)
    """
    with open(path, "rb") as f:
        config = tomllib.load(f)
    
    training_cfg = TrainingCfg(**config.get("training", {}))
    model_cfg = ModelCfg(**config.get("model", {}))
    algo_cfg = AlgoCfg(**config.get("algorithm", {}))
    eval_cfg = EvalCfg(**config.get("evaluation", {}))
    
    return training_cfg, model_cfg, algo_cfg, eval_cfg


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description="Train offline RL algorithms")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to TOML configuration file"
    )
    parser.add_argument(
        "--algo",
        type=str,
        help="Algorithm name (overrides config)"
    )
    parser.add_argument(
        "--env",
        type=str,
        help="Environment name (overrides config)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        help="WandB mode (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load config
    training_cfg, model_cfg, algo_cfg, eval_cfg = load_config_from_toml(args.config)
    
    # Apply CLI overrides
    if args.algo:
        training_cfg = TrainingCfg(**{**asdict(training_cfg), "algo": args.algo})
    if args.env:
        training_cfg = TrainingCfg(**{**asdict(training_cfg), "env_name": args.env})
    if args.seed is not None:
        training_cfg = TrainingCfg(**{**asdict(training_cfg), "seed": args.seed})
    if args.wandb_mode:
        training_cfg = TrainingCfg(**{**asdict(training_cfg), "wandb_mode": args.wandb_mode})
    
    # Print configuration
    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(f"Algorithm: {training_cfg.algo}")
    print(f"Environment: {training_cfg.env_name}")
    print(f"Seed: {training_cfg.seed}")
    print(f"Total steps: {training_cfg.total_steps:,}")
    print(f"Batch size: {training_cfg.batch_size}")
    print(f"Learning rate: {training_cfg.lr}")
    print("=" * 80)
    
    # Create and run experiment
    runner = ExperimentRunner(training_cfg, model_cfg, algo_cfg, eval_cfg)
    runner.run()


if __name__ == "__main__":
    main()
