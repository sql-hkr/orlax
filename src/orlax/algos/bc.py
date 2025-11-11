"""Behavior Cloning (BC) algorithm implementation."""

from typing import Any, Dict
from functools import partial
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from ..core.types import Batch
from ..core.cfg import TrainingCfg, ModelCfg, AlgoCfg
from ..models.nets import Actor


class BC:
    """Behavior Cloning for offline RL.
    
    Trains a policy via supervised learning to minimize:
    
    $$L = \mathbb{E}_{(s,a) \sim \mathcal{D}}[(\pi(s) - a)^2]$$
    
    BC serves as a simple baseline and is effective when the dataset contains
    high-quality demonstrations without distribution shift.
    
    Reference: Pomerleau, "ALVINN: An Autonomous Land Vehicle in a Neural Network" (1989)
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cfg: TrainingCfg,
        model_cfg: ModelCfg,
        algo_cfg: AlgoCfg,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.algo_cfg = algo_cfg
        
        # Initialize actor network
        self.actor = Actor(
            action_dim=act_dim,
            hidden_dims=model_cfg.hidden_dims,
            activation=model_cfg.activation,
        )
    
    def init(self, rng: jnp.ndarray) -> Dict[str, Any]:
        """Initialize algorithm state.
        
        Args:
            rng: JAX random key
            
        Returns:
            Dictionary containing actor train state, step counter, and RNG key.
        """
        rng, actor_rng = jax.random.split(rng)
        
        dummy_obs = jnp.zeros((1, self.obs_dim))
        actor_params = self.actor.init(actor_rng, dummy_obs)
        actor_tx = optax.adam(learning_rate=self.cfg.lr)
        
        actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=actor_tx,
        )
        
        return {
            "actor": actor_state,
            "step": 0,
            "rng": rng,
        }
    
    @partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: Dict[str, Any],
        batch: Batch,
        rng: jnp.ndarray,
    ) -> tuple[Dict[str, Any], Dict[str, float]]:
        """Perform one training step.
        
        Minimizes $L = \mathbb{E}[(\pi(s) - a)^2]$ via gradient descent.
        
        Args:
            state: Current algorithm state
            batch: Training batch
            rng: Random key
            
        Returns:
            Tuple of (new_state, metrics)
        """
        rng = jax.random.split(rng)[0]
        
        def actor_loss_fn(actor_params):
            pred_actions, _ = state["actor"].apply_fn(actor_params, batch.obs)
            squared_errors = (pred_actions - batch.act) ** 2
            mse_loss = jnp.mean(squared_errors)
            
            info = {
                "action_mean": jnp.mean(jnp.abs(pred_actions)),
                "action_std": jnp.std(pred_actions),
                "target_action_mean": jnp.mean(jnp.abs(batch.act)),
            }
            
            return mse_loss, info
        
        (actor_loss, actor_info), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(state["actor"].params)
        
        actor_state = state["actor"].apply_gradients(grads=actor_grads)
        
        new_state = {
            "actor": actor_state,
            "step": state["step"] + 1,
            "rng": rng,
        }
        
        metrics = {
            "actor_loss": actor_loss,
            "mse_loss": actor_loss,
            "action_mean": actor_info["action_mean"],
            "action_std": actor_info["action_std"],
            "target_action_mean": actor_info["target_action_mean"],
        }
        
        return new_state, metrics
    
    def eval_step(self, state: Dict[str, Any], obs: jnp.ndarray) -> jnp.ndarray:
        """Get deterministic action for evaluation.
        
        Args:
            state: Algorithm state
            obs: Observation [obs_dim] or [batch_size, obs_dim]
            
        Returns:
            Action [act_dim] or [batch_size, act_dim]
        """
        mean, _ = state["actor"].apply_fn(state["actor"].params, obs)
        return mean
