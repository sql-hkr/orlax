"""Policy and value network architectures."""

from dataclasses import dataclass
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


class ConditionalVAE(nn.Module):
    """Conditional Variational Autoencoder (CVAE) for modeling p(a|s).
    
    Used by BCQ and PLAS to constrain actions to the behavioral data distribution.
    
    Attributes:
        action_dim: Action dimension
        latent_dim: Latent space dimension
        hidden_dims: Hidden layer dimensions
        activation: Activation function
        max_action: Maximum action magnitude
    """
    
    action_dim: int
    latent_dim: int = 64
    hidden_dims: Sequence[int] = (256, 256)
    activation: str = "relu"
    max_action: float = 1.0
    
    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        training: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Encode state-action pair to latent distribution."""
        x = jnp.concatenate([obs, action], axis=-1)
        h = MLP(hidden_dims=self.hidden_dims, activation=self.activation, name="encoder")(x, training=training)
        mean = nn.Dense(self.latent_dim, name="latent_mean")(h)
        log_std = nn.Dense(self.latent_dim, name="latent_log_std")(h)
        log_std = jnp.clip(log_std, -4.0, 2.0)
        
        # Initialize decoder parameters in the same pass
        dummy_z = jnp.zeros((obs.shape[0], self.latent_dim))
        _ = self.decode(obs, dummy_z, training=training)
        
        return mean, log_std
    
    @nn.compact
    def decode(
        self,
        obs: jnp.ndarray,
        z: jnp.ndarray,
        training: bool = False
    ) -> jnp.ndarray:
        """Decode latent vector and state into action."""
        x = jnp.concatenate([obs, z], axis=-1)
        h = MLP(hidden_dims=self.hidden_dims, activation=self.activation, name="decoder")(x, training=training)
        action = nn.Dense(self.action_dim, name="action_head")(h)
        return nn.tanh(action) * self.max_action


class EnsembleDynamics(nn.Module):
    """Ensemble dynamics model for model-based offline RL.
    
    Predicts next observation change (delta_obs) and reward given (obs, action).
    Used by MOPO, RAMBO, COMBO, MOBILE, and BREMEN.
    
    Attributes:
        obs_dim: Observation dimension
        num_models: Number of dynamics models in ensemble
        hidden_dims: Hidden layer dimensions
        activation: Activation function
    """
    
    obs_dim: int
    num_models: int = 4
    hidden_dims: Sequence[int] = (256, 256)
    activation: str = "relu"
    
    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        training: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass predicting delta_obs and rewards across ensemble.
        
        Args:
            obs: Observations [batch_size, obs_dim]
            action: Actions [batch_size, act_dim]
            training: Training mode
            
        Returns:
            delta_obs: [batch_size, num_models, obs_dim]
            rewards: [batch_size, num_models]
        """
        x = jnp.concatenate([obs, action], axis=-1)
        
        delta_obs_list = []
        reward_list = []
        for i in range(self.num_models):
            h = MLP(
                hidden_dims=self.hidden_dims,
                activation=self.activation,
                name=f"dynamics_{i}"
            )(x, training=training)
            delta = nn.Dense(self.obs_dim, name=f"delta_{i}")(h)
            rew = nn.Dense(1, name=f"reward_{i}")(h)
            
            delta_obs_list.append(delta)
            reward_list.append(rew.squeeze(-1))
            
        return jnp.stack(delta_obs_list, axis=1), jnp.stack(reward_list, axis=1)


@dataclass(frozen=True)
class DiffusionSchedule:
    """Precomputed schedule parameters for DDPM diffusion processes.
    
    Attributes:
        betas: Variance schedule beta_t
        alphas: alpha_t = 1 - beta_t
        alphas_cumprod: Cumulative product of alphas
        alphas_cumprod_prev: Shifted cumulative product of alphas
        sqrt_alphas_cumprod: Square root of cumulative alphas
        sqrt_one_minus_alphas_cumprod: Square root of (1 - cumulative alphas)
        posterior_variance: Posterior variance for reverse sampling
    """
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alphas_cumprod: jnp.ndarray
    alphas_cumprod_prev: jnp.ndarray
    sqrt_alphas_cumprod: jnp.ndarray
    sqrt_one_minus_alphas_cumprod: jnp.ndarray
    posterior_variance: jnp.ndarray

    @classmethod
    def make(cls, n_steps: int = 5, beta_start: float = 1e-4, beta_end: float = 0.05) -> "DiffusionSchedule":
        """Create linear beta schedule and derived DDPM constants."""
        betas = jnp.linspace(beta_start, beta_end, n_steps)
        alphas = 1.0 - betas
        alphas_cumprod = jnp.cumprod(alphas)
        alphas_cumprod_prev = jnp.concatenate([jnp.array([1.0]), alphas_cumprod[:-1]])
        sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - alphas_cumprod)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / jnp.maximum(1.0 - alphas_cumprod, 1e-8)
        return cls(
            betas=betas,
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            alphas_cumprod_prev=alphas_cumprod_prev,
            sqrt_alphas_cumprod=sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            posterior_variance=posterior_variance,
        )


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""
    dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        half_dim = self.dim // 2
        emb = jnp.log(10000.0) / jnp.maximum(half_dim - 1, 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
        emb = x[:, None] * emb[None, :]
        return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)


class DiffusionModel(nn.Module):
    """Conditional diffusion MLP network predicting noise epsilon given state, noisy action, and time.
    
    Used by Diffusion-QL and EDP.
    
    Attributes:
        action_dim: Action space dimension
        hidden_dims: Hidden layer dimensions
        activation: Activation function
        time_dim: Dimension of sinusoidal time embeddings
        max_action: Action boundary magnitude
    """
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)
    activation: str = "relu"
    time_dim: int = 16
    max_action: float = 1.0

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        time: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        """Forward pass predicting noise epsilon.
        
        Args:
            obs: Observations [batch_size, obs_dim]
            action: Noisy actions [batch_size, action_dim]
            time: Timestep indices or scalar times [batch_size]
            training: Whether in training mode
            
        Returns:
            Predicted noise [batch_size, action_dim]
        """
        t_flat = jnp.reshape(time, (-1,))
        t_emb = SinusoidalPosEmb(self.time_dim)(t_flat)
        t_emb = MLP(
            hidden_dims=(self.time_dim * 2,),
            activation=self.activation,
            name="time_mlp",
        )(t_emb, training=training)

        x = jnp.concatenate([obs, action, t_emb], axis=-1)
        h = MLP(
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            name="trunk",
        )(x, training=training)

        return nn.Dense(self.action_dim, name="output_head")(h)

