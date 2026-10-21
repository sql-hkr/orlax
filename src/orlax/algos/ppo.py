"""Proximal Policy Optimization (PPO) algorithm implementation."""

from typing import Any, Dict
from functools import partial
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
import distrax

from ..core.types import Batch
from ..core.cfg import TrainingCfg, ModelCfg, AlgoCfg
from ..models.nets import Actor, ValueNetwork


class PPO:
    r"""Proximal Policy Optimization for continuous control.
    
    PPO optimizes a clipped surrogate objective that prevents destructively large
    policy updates:
    
    $$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]$$
    
    where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$.
    
    Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
    https://arxiv.org/abs/1707.06347
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
        
        self.actor = Actor(
            action_dim=act_dim,
            hidden_dims=model_cfg.hidden_dims,
            activation=model_cfg.activation,
        )
        
        self.value = ValueNetwork(
            hidden_dims=model_cfg.hidden_dims,
            activation=model_cfg.activation,
        )
    
    def init(self, rng: jnp.ndarray) -> Dict[str, Any]:
        """Initialize algorithm state.
        
        Args:
            rng: JAX random key
            
        Returns:
            Dictionary containing actor, value train states, step counter, and RNG key.
        """
        rng, actor_rng, value_rng = jax.random.split(rng, 3)
        
        dummy_obs = jnp.zeros((1, self.obs_dim))
        
        actor_params = self.actor.init(actor_rng, dummy_obs)
        value_params = self.value.init(value_rng, dummy_obs)
        
        actor_tx = optax.adam(learning_rate=self.cfg.lr)
        value_tx = optax.adam(learning_rate=self.cfg.lr)
        
        actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=actor_tx,
        )
        
        value_state = TrainState.create(
            apply_fn=self.value.apply,
            params=value_params,
            tx=value_tx,
        )
        
        return {
            "actor": actor_state,
            "value": value_state,
            "step": 0,
            "rng": rng,
        }
    
    def _sample_actions(
        self,
        actor_params: Any,
        obs: jnp.ndarray,
        rng: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample actions and compute log probabilities.
        
        Args:
            actor_params: Actor network parameters
            obs: Observations [batch_size, obs_dim]
            rng: Random key
            
        Returns:
            Tuple of (actions, log_probs)
        """
        mean, log_std = self.actor.apply(actor_params, obs)
        std = jnp.exp(log_std)
        
        dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
        actions = dist.sample(seed=rng)
        log_probs = dist.log_prob(actions)
        
        return actions, log_probs
    
    @partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: Dict[str, Any],
        batch: Batch,
        rng: jnp.ndarray,
    ) -> tuple[Dict[str, Any], Dict[str, float]]:
        """Perform one PPO training step.
        
        Updates:
            1. Value network by minimizing TD error against target returns
            2. Policy network with clipped surrogate objective and entropy bonus
            
        Args:
            state: Current algorithm state
            batch: Training batch
            rng: Random key
            
        Returns:
            Tuple of (new_state, metrics)
        """
        rng = jax.random.split(rng)[0]
        
        # 1. Compute target values and normalized advantages
        target_v = state["value"].apply_fn(state["value"].params, batch.next_obs)
        current_v = state["value"].apply_fn(state["value"].params, batch.obs)
        
        td_target = batch.reward + self.algo_cfg.gamma * (1.0 - batch.done) * target_v
        advantages = td_target - current_v
        
        # Normalize advantages for stability
        adv_norm = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
        adv_norm = jax.lax.stop_gradient(adv_norm)
        td_target = jax.lax.stop_gradient(td_target)
        
        # 2. Compute old policy action log probabilities
        old_mean, old_log_std = state["actor"].apply_fn(state["actor"].params, batch.obs)
        old_dist = distrax.MultivariateNormalDiag(loc=old_mean, scale_diag=jnp.exp(old_log_std))
        old_log_probs = jax.lax.stop_gradient(old_dist.log_prob(batch.act))
        
        # 3. Value network loss and update
        def value_loss_fn(value_params):
            pred_v = state["value"].apply_fn(value_params, batch.obs)
            value_loss = 0.5 * jnp.mean((pred_v - td_target) ** 2)
            
            info = {
                "v_mean": jnp.mean(pred_v),
                "v_std": jnp.std(pred_v),
                "target_v_mean": jnp.mean(td_target),
            }
            return value_loss, info
        
        (value_loss, value_info), value_grads = jax.value_and_grad(
            value_loss_fn, has_aux=True
        )(state["value"].params)
        value_state = state["value"].apply_gradients(grads=value_grads)
        
        # 4. Actor network clipped surrogate loss and update
        def actor_loss_fn(actor_params):
            mean, log_std = state["actor"].apply_fn(actor_params, batch.obs)
            dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
            new_log_probs = dist.log_prob(batch.act)
            
            # Probability ratio: r_t(theta) = exp(log pi_new - log pi_old)
            ratio = jnp.exp(new_log_probs - old_log_probs)
            surr1 = ratio * adv_norm
            surr2 = (
                jnp.clip(ratio, 1.0 - self.algo_cfg.clip_eps, 1.0 + self.algo_cfg.clip_eps)
                * adv_norm
            )
            
            policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
            entropy = jnp.mean(dist.entropy())
            actor_loss = policy_loss - self.algo_cfg.entropy_coef * entropy
            
            # Diagnostics
            approx_kl = jnp.mean(old_log_probs - new_log_probs)
            clip_fraction = jnp.mean(jnp.abs(ratio - 1.0) > self.algo_cfg.clip_eps)
            
            info = {
                "policy_loss": policy_loss,
                "entropy": entropy,
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
                "ratio_mean": jnp.mean(ratio),
            }
            return actor_loss, info
        
        (actor_loss, actor_info), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(state["actor"].params)
        actor_state = state["actor"].apply_gradients(grads=actor_grads)
        
        new_state = {
            "actor": actor_state,
            "value": value_state,
            "step": state["step"] + 1,
            "rng": rng,
        }
        
        metrics = {
            "value_loss": value_loss,
            "actor_loss": actor_loss,
            "policy_loss": actor_info["policy_loss"],
            "entropy": actor_info["entropy"],
            "approx_kl": actor_info["approx_kl"],
            "clip_fraction": actor_info["clip_fraction"],
            "ratio_mean": actor_info["ratio_mean"],
            "v_mean": value_info["v_mean"],
            "advantage_mean": jnp.mean(advantages),
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
