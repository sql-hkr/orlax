"""Configuration dataclasses for typed, autocompletion-friendly configs.

Use these dataclasses throughout the project instead of plain dictionaries.
This improves type safety and editor autocompletion.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TrainingCfg:
    """Main training configuration.
    
    Args:
        algo: Algorithm name ('cql', 'iql')
        env_name: Environment name (e.g., 'halfcheetah-medium-v2')
        batch_size: Batch size for training
        total_steps: Total training steps
        lr: Learning rate
        seed: Random seed for reproducibility
        device: Device to use ('cpu', 'cuda:0', or None for auto)
        eval_freq: Evaluation frequency in steps
        checkpoint_freq: Checkpoint save frequency in steps
        log_freq: Logging frequency in steps
        wandb_project: WandB project name
        wandb_entity: WandB entity (username or team)
        wandb_mode: WandB mode ('online', 'offline', 'disabled')
    """
    algo: str
    env_name: str
    batch_size: int = 256
    total_steps: int = 1_000_000
    lr: float = 3e-4
    seed: int = 0
    device: Optional[str] = None
    eval_freq: int = 5_000
    checkpoint_freq: int = 50_000
    log_freq: int = 1_000
    wandb_project: str = "orlax"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"


@dataclass(frozen=True)
class ModelCfg:
    """Model architecture configuration.
    
    Args:
        hidden_dims: List of hidden layer dimensions
        activation: Activation function name ('relu', 'tanh', 'gelu')
        dropout_rate: Dropout rate (0.0 to disable)
        layer_norm: Whether to use layer normalization
        use_bias: Whether to use bias in linear layers
    """
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    activation: str = "relu"
    dropout_rate: float = 0.0
    layer_norm: bool = False
    use_bias: bool = True


@dataclass(frozen=True)
class AlgoCfg:
    """Algorithm-specific hyperparameters.
    
    Each algorithm can override these defaults with their specific parameters.
    
    Args:
        gamma: Discount factor
        tau: Target network soft update rate
        alpha: Conservative penalty weight (CQL)
        beta: Temperature parameter (IQL)
        expectile: Expectile parameter (IQL)
        use_automatic_entropy_tuning: Auto-tune entropy coefficient
        n_critics: Number of critic networks
    """
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 1.0
    beta: float = 3.0
    expectile: float = 0.7
    use_automatic_entropy_tuning: bool = True
    n_critics: int = 2


@dataclass(frozen=True)
class EvalCfg:
    """Evaluation configuration.
    
    Args:
        num_episodes: Number of episodes to evaluate
        deterministic: Use deterministic policy for evaluation
        render: Render episodes during evaluation
        save_video: Save evaluation videos
        moving_avg_window: Window size for moving average of eval rewards
    """
    num_episodes: int = 10
    deterministic: bool = True
    render: bool = False
    save_video: bool = False
    moving_avg_window: int = 100
