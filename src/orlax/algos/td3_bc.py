"""TD3+BC (Twin Delayed DDPG + Behavioral Cloning) algorithm implementation."""

from typing import Any, Dict
from functools import partial
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from ..core.types import Batch
from ..core.cfg import TrainingCfg, ModelCfg, AlgoCfg
from ..models.nets import Actor, Critic


class TD3BC:
    r"""TD3+BC for offline reinforcement learning.
    
    A minimalist offline RL algorithm that adds a simple behavioral cloning (BC)
    regularization term to the TD3 actor loss:
    
    $$\pi = \arg\max_\pi \mathbb{E}_{(s, a) \sim \mathcal{D}} [\lambda Q(s, \pi(s)) - (\pi(s) - a)^2]$$
    
    where the normalization factor $\lambda$ is:
    
    $$\lambda = \frac{\alpha}{\frac{1}{N}\sum_{(s, a)} |Q(s, \pi(s))|}$$
    
    This balances policy improvement against staying close to the dataset distribution
    without complex density models or out-of-distribution penalties.
    
    Reference: Fujimoto & Gu, "A Minimalist Approach to Offline Reinforcement Learning" (NeurIPS 2021)
    https://arxiv.org/abs/2106.06860
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
        
        # In TD3+BC, default alpha is 2.5 if not specifically configured
        self.alpha = algo_cfg.alpha if algo_cfg.alpha != 1.0 else 2.5
        
        self.actor = Actor(
            action_dim=act_dim,
            hidden_dims=model_cfg.hidden_dims,
            activation=model_cfg.activation,
        )
        
        self.critic = Critic(
            hidden_dims=model_cfg.hidden_dims,
            activation=model_cfg.activation,
            num_critics=algo_cfg.n_critics,
        )
    
    def init(self, rng: jnp.ndarray) -> Dict[str, Any]:
        """Initialize algorithm state.
        
        Args:
            rng: JAX random key
            
        Returns:
            Dictionary containing actor, critic, target actor, target critic, step counter, and RNG key.
        """
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        
        dummy_obs = jnp.zeros((1, self.obs_dim))
        dummy_act = jnp.zeros((1, self.act_dim))
        
        actor_params = self.actor.init(actor_rng, dummy_obs)
        critic_params = self.critic.init(critic_rng, dummy_obs, dummy_act)
        
        actor_tx = optax.adam(learning_rate=self.cfg.lr)
        critic_tx = optax.adam(learning_rate=self.cfg.lr)
        
        actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=actor_tx,
        )
        
        critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=critic_tx,
        )
        
        target_actor_params = jax.tree.map(lambda x: x, actor_params)
        target_critic_params = jax.tree.map(lambda x: x, critic_params)
        
        return {
            "actor": actor_state,
            "critic": critic_state,
            "target_actor": target_actor_params,
            "target_critic": target_critic_params,
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
        """Perform one TD3+BC training step.
        
        Updates:
            1. Twin Q-critics using TD learning with target policy smoothing
            2. Policy network every `policy_freq` steps via Q-maximization + BC loss
            3. Target networks via Polyak averaging every `policy_freq` steps
            
        Args:
            state: Current algorithm state
            batch: Training batch
            rng: Random key
            
        Returns:
            Tuple of (new_state, metrics)
        """
        rng, critic_rng = jax.random.split(rng)
        
        # 1. Critic update with target policy smoothing
        def critic_loss_fn(critic_params):
            next_mean, _ = self.actor.apply(state["target_actor"], batch.next_obs)
            
            # Target policy smoothing noise
            noise = jax.random.normal(critic_rng, next_mean.shape) * self.algo_cfg.target_noise
            noise = jnp.clip(noise, -self.algo_cfg.noise_clip, self.algo_cfg.noise_clip)
            next_actions = jnp.clip(next_mean + noise, -1.0, 1.0)
            
            target_q = state["critic"].apply_fn(
                state["target_critic"],
                batch.next_obs,
                next_actions,
            )
            target_q = jnp.min(target_q, axis=-1)
            
            td_target = batch.reward + self.algo_cfg.gamma * (1.0 - batch.done) * target_q
            td_target = jax.lax.stop_gradient(td_target)
            
            q_values = state["critic"].apply_fn(
                critic_params,
                batch.obs,
                batch.act,
            )
            
            critic_loss = jnp.mean((q_values - td_target[:, None]) ** 2)
            
            info = {
                "q_mean": jnp.mean(q_values),
                "q_std": jnp.std(q_values),
                "target_q_mean": jnp.mean(td_target),
            }
            return critic_loss, info
        
        (critic_loss, critic_info), critic_grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )(state["critic"].params)
        critic_state = state["critic"].apply_gradients(grads=critic_grads)
        
        # 2. Delayed policy update (every `policy_freq` steps)
        def update_actor_and_targets():
            def actor_loss_fn(actor_params):
                pred_actions, _ = state["actor"].apply_fn(actor_params, batch.obs)
                q_values = state["critic"].apply_fn(
                    critic_state.params,
                    batch.obs,
                    pred_actions,
                )
                q1 = q_values[:, 0]
                
                # Normalization factor lambda = alpha / mean(|Q|)
                lmbda = self.alpha / (jax.lax.stop_gradient(jnp.mean(jnp.abs(q1))) + 1e-8)
                
                q_loss = -jnp.mean(q1)
                bc_loss = jnp.mean((pred_actions - batch.act) ** 2)
                actor_loss = lmbda * q_loss + bc_loss
                
                info = {
                    "actor_loss": actor_loss,
                    "q_loss": q_loss,
                    "bc_loss": bc_loss,
                    "lmbda": lmbda,
                }
                return actor_loss, info
            
            (actor_loss, actor_info), actor_grads = jax.value_and_grad(
                actor_loss_fn, has_aux=True
            )(state["actor"].params)
            new_actor_state = state["actor"].apply_gradients(grads=actor_grads)
            
            # Polyak averaging target updates
            new_target_critic = jax.tree.map(
                lambda target, online: self.algo_cfg.tau * online + (1.0 - self.algo_cfg.tau) * target,
                state["target_critic"],
                critic_state.params,
            )
            new_target_actor = jax.tree.map(
                lambda target, online: self.algo_cfg.tau * online + (1.0 - self.algo_cfg.tau) * target,
                state["target_actor"],
                new_actor_state.params,
            )
            return new_actor_state, new_target_critic, new_target_actor, actor_info
        
        def skip_actor_and_targets():
            dummy_info = {
                "actor_loss": 0.0,
                "q_loss": 0.0,
                "bc_loss": 0.0,
                "lmbda": 0.0,
            }
            return state["actor"], state["target_critic"], state["target_actor"], dummy_info
        
        should_update_actor = (state["step"] % self.algo_cfg.policy_freq == 0)
        actor_state, target_critic, target_actor, actor_info = jax.lax.cond(
            should_update_actor,
            update_actor_and_targets,
            skip_actor_and_targets,
        )
        
        new_state = {
            "actor": actor_state,
            "critic": critic_state,
            "target_actor": target_actor,
            "target_critic": target_critic,
            "step": state["step"] + 1,
            "rng": rng,
        }
        
        metrics = {
            "critic_loss": critic_loss,
            "actor_loss": actor_info["actor_loss"],
            "bc_loss": actor_info["bc_loss"],
            "q_loss": actor_info["q_loss"],
            "lmbda": actor_info["lmbda"],
            "q_mean": critic_info["q_mean"],
            "q_std": critic_info["q_std"],
            "target_q_mean": critic_info["target_q_mean"],
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


# Alias
TD3_BC = TD3BC
