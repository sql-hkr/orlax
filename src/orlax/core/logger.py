"""Logging utilities with WandB integration and tqdm progress bars."""

from typing import Any, Optional
import wandb
from tqdm import tqdm


class Logger:
    """Logger wrapper for WandB and console output.
    
    Provides a unified interface for logging metrics to WandB and displaying
    progress in the terminal with tqdm.
    
    Example:
        logger = Logger(config, project="orlax", entity="myteam")
        logger.log({"loss": 0.5}, step=100)
        logger.update_progress({"eval_reward": 500.0})
    """
    
    def __init__(
        self,
        config: Any,
        project: str = "orlax",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
        mode: str = "online",
    ):
        """Initialize logger with WandB.
        
        Args:
            config: Configuration object (dataclass) to log
            project: WandB project name
            entity: WandB entity (username or team)
            name: Run name (auto-generated if None)
            group: WandB group for organizing runs
            mode: WandB mode ('online', 'offline', 'disabled')
        """
        self.config = config
        self.mode = mode
        
        # Convert dataclass to dict for WandB
        config_dict = self._to_wandb_dict(config)
        
        # Initialize WandB
        wandb.init(
            project=project,
            entity=entity,
            name=name,
            group=group,
            config=config_dict,
            mode=mode,
        )
        
        self.pbar: Optional[tqdm] = None
    
    def _to_wandb_dict(self, obj: Any) -> dict:
        """Convert dataclass config to flat dict for WandB."""
        if hasattr(obj, "__dataclass_fields__"):
            return {
                k: self._to_wandb_dict(v) 
                for k, v in obj.__dict__.items()
            }
        elif isinstance(obj, (list, tuple)):
            return [self._to_wandb_dict(item) for item in obj]
        else:
            return obj
    
    def init_progress(self, total: int, desc: str = "Training"):
        """Initialize tqdm progress bar.
        
        Args:
            total: Total number of steps
            desc: Progress bar description
        """
        self.pbar = tqdm(total=total, desc=desc)
    
    def log(self, metrics: dict[str, Any], step: Optional[int] = None, commit: bool = True):
        """Log metrics to WandB.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Training step (optional)
            commit: Whether to commit the log immediately
        """
        if self.mode != "disabled":
            wandb.log(metrics, step=step, commit=commit)
    
    def update_progress(self, postfix: Optional[dict] = None, n: int = 1):
        """Update progress bar.
        
        Args:
            postfix: Dictionary to display in progress bar
            n: Number of steps to increment
        """
        if self.pbar is not None:
            self.pbar.update(n)
            if postfix is not None:
                self.pbar.set_postfix(postfix)
    
    def close_progress(self):
        """Close progress bar."""
        if self.pbar is not None:
            self.pbar.close()
    
    def save_artifact(self, path: str, name: str, artifact_type: str = "model"):
        """Save artifact to WandB.
        
        Args:
            path: Path to artifact file
            name: Artifact name
            artifact_type: Type of artifact ('model', 'dataset', etc.)
        """
        if self.mode != "disabled":
            artifact = wandb.Artifact(name, type=artifact_type)
            artifact.add_file(path)
            wandb.log_artifact(artifact)
    
    def finish(self):
        """Finish WandB run."""
        if self.mode != "disabled":
            wandb.finish()
