"""Policy and value network architectures."""

from typing import Sequence
import jax.numpy as jnp
from flax import linen as nn

from .mlp import MLP


class Actor(nn.Module):
    """Actor network for continuous control.
    
    Outputs mean and log_std for Gaussian policy.
    
    Attributes:
        action_dim: Action space dimension
        hidden_dims: Hidden layer dimensions
        activation: Activation function
        log_std_min: Minimum log standard deviation
        log_std_max: Maximum log standard deviation
        state_dependent_std: Whether std depends on state
        max_action: Maximum action value (default 1.0 for normalized actions)
        
    Example:
        actor = Actor(action_dim=6, hidden_dims=(256, 256))
        mean, log_std = actor(obs)
        action = mean + jax.random.normal(key, mean.shape) * jnp.exp(log_std)
    """
    
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)
    activation: str = "relu"
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    state_dependent_std: bool = True
    max_action: float = 1.0
    
    @nn.compact
    def __call__(self, obs: jnp.ndarray, training: bool = False) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.
        
        Args:
            obs: Observations [batch_size, obs_dim]
            training: Whether in training mode
            
        Returns:
            Tuple of (mean, log_std) for Gaussian policy
        """
        # Shared trunk
        x = MLP(
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            name="trunk"
        )(obs, training=training)
        
        # Mean head
        mean = nn.Dense(self.action_dim, name="mean")(x)
        mean = nn.tanh(mean) * self.max_action  # Bound actions to [-max_action, max_action]
        
        # Log std head
        if self.state_dependent_std:
            log_std = nn.Dense(self.action_dim, name="log_std")(x)
        else:
            log_std = self.param(
                "log_std",
                nn.initializers.zeros,
                (self.action_dim,)
            )
            log_std = jnp.broadcast_to(log_std, mean.shape)
        
        # Clip log std for stability
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std


class Critic(nn.Module):
    """Critic network (Q-function) for continuous control.
    
    Attributes:
        hidden_dims: Hidden layer dimensions
        activation: Activation function
        num_critics: Number of critic heads (for ensembles)
        
    Example:
        critic = Critic(hidden_dims=(256, 256), num_critics=2)
        q_values = critic(obs, action)  # shape: [batch_size, num_critics]
    """
    
    hidden_dims: Sequence[int] = (256, 256)
    activation: str = "relu"
    num_critics: int = 2
    
    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        training: bool = False
    ) -> jnp.ndarray:
        """Forward pass.
        
        Args:
            obs: Observations [batch_size, obs_dim]
            action: Actions [batch_size, action_dim]
            training: Whether in training mode
            
        Returns:
            Q-values [batch_size, num_critics]
        """
        # Concatenate obs and action
        x = jnp.concatenate([obs, action], axis=-1)
        
        # Multiple critic heads
        q_values = []
        for i in range(self.num_critics):
            q = MLP(
                hidden_dims=self.hidden_dims,
                output_dim=1,
                activation=self.activation,
                name=f"critic_{i}"
            )(x, training=training)
            q_values.append(q.squeeze(-1))
        
        return jnp.stack(q_values, axis=-1)


class ValueNetwork(nn.Module):
    """Value network (V-function) for policy evaluation.
    
    Attributes:
        hidden_dims: Hidden layer dimensions
        activation: Activation function
        
    Example:
        value_net = ValueNetwork(hidden_dims=(256, 256))
        value = value_net(obs)
    """
    
    hidden_dims: Sequence[int] = (256, 256)
    activation: str = "relu"
    
    @nn.compact
    def __call__(self, obs: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass.
        
        Args:
            obs: Observations [batch_size, obs_dim]
            training: Whether in training mode
            
        Returns:
            State values [batch_size]
        """
        x = MLP(
            hidden_dims=self.hidden_dims,
            output_dim=1,
            activation=self.activation,
            name="value"
        )(obs, training=training)
        
        return x.squeeze(-1)

