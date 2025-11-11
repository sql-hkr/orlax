"""Conservative Q-Learning (CQL) algorithm implementation."""

from typing import Any, Dict
from functools import partial
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
import distrax

from ..core.types import Batch
from ..core.cfg import TrainingCfg, ModelCfg, AlgoCfg
from ..models.nets import Actor, Critic


class CQL:
    """Conservative Q-Learning for offline RL.
    
    Learns conservative Q-functions by penalizing Q-values on out-of-distribution
    actions while encouraging higher Q-values on dataset actions:
    
    $$L_{CQL} = \mathbb{E}[\log \sum_a \exp Q(s,a)] - \mathbb{E}_{(s,a)\sim\mathcal{D}}[Q(s,a)]$$
    
    Combined with SAC-style policy optimization and twin Q-networks.
    
    Reference: Kumar et al., "Conservative Q-Learning for Offline RL" (2020)
    https://arxiv.org/abs/2006.04779
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
        self.model_cfg = algo_cfg
        
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
        
        self.target_entropy = -act_dim
    
    def init(self, rng: jnp.ndarray) -> Dict[str, Any]:
        """Initialize algorithm state.
        
        Args:
            rng: JAX random key
            
        Returns:
            Dictionary containing actor, critic, target critic, log_alpha, optimizer state, step counter, and RNG key.
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
        
        target_critic_params = jax.tree.map(lambda x: x, critic_params)
        log_alpha = jnp.array(0.0)
        
        if self.algo_cfg.use_automatic_entropy_tuning:
            self.alpha_tx = optax.adam(learning_rate=self.cfg.lr)
            alpha_opt_state = self.alpha_tx.init(log_alpha)
        else:
            self.alpha_tx = None
            alpha_opt_state = None
        
        return {
            "actor": actor_state,
            "critic": critic_state,
            "target_critic": target_critic_params,
            "log_alpha": log_alpha,
            "alpha_opt_state": alpha_opt_state,
            "step": 0,
            "rng": rng,
        }
    
    def _sample_actions(
        self,
        actor_params: Any,
        obs: jnp.ndarray,
        rng: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample actions using reparameterization trick.
        
        Args:
            actor_params: Actor network parameters
            obs: Observations [batch_size, obs_dim]
            rng: Random key
            
        Returns:
            Tuple of (actions, log_probs) with shapes [batch_size, act_dim] and [batch_size]
        """
        mean, log_std = self.actor.apply(actor_params, obs)
        std = jnp.exp(log_std)
        
        dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
        actions = dist.sample(seed=rng)
        log_probs = dist.log_prob(actions)
        
        return actions, log_probs
    
    def _compute_cql_penalty(
        self,
        critic_params: Any,
        actor_params: Any,
        obs: jnp.ndarray,
        actions_data: jnp.ndarray,
        rng: jnp.ndarray,
        n_samples: int = 10,
    ) -> jnp.ndarray:
        """Compute CQL conservative penalty.
        
        Computes $\log \sum_a \exp Q(s,a) - Q(s,a_{data})$ by sampling actions.
        
        Args:
            critic_params: Critic network parameters
            actor_params: Actor network parameters
            obs: Observations [batch_size, obs_dim]
            actions_data: Dataset actions [batch_size, act_dim]
            rng: Random key
            n_samples: Number of actions to sample per state
            
        Returns:
            CQL penalty (scalar)
        """
        batch_size = obs.shape[0]
        
        obs_tiled = jnp.tile(obs[:, None, :], (1, n_samples, 1))
        obs_tiled = obs_tiled.reshape(-1, self.obs_dim)
        
        rng_random, rng_policy = jax.random.split(rng)
        random_actions = jax.random.uniform(
            rng_random,
            shape=(batch_size * n_samples, self.act_dim),
            minval=-1.0,
            maxval=1.0,
        )
        
        policy_actions, _ = self._sample_actions(
            actor_params,
            obs_tiled,
            rng_policy,
        )
        
        q_random = self.critic.apply(critic_params, obs_tiled, random_actions)
        q_policy = self.critic.apply(critic_params, obs_tiled, policy_actions)
        
        q_random = q_random.reshape(batch_size, n_samples, -1)
        q_policy = q_policy.reshape(batch_size, n_samples, -1)
        
        q_sampled = jnp.concatenate([q_random, q_policy], axis=1)
        q_logsumexp = jax.scipy.special.logsumexp(q_sampled, axis=1)
        
        q_data = self.critic.apply(critic_params, obs, actions_data)
        
        cql_penalty = jnp.mean(q_logsumexp - q_data)
        
        return cql_penalty
    
    @partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: Dict[str, Any],
        batch: Batch,
        rng: jnp.ndarray,
    ) -> tuple[Dict[str, Any], Dict[str, float]]:
        """Perform one CQL training step.
        
        Updates: (1) Q-functions with TD loss + CQL penalty, (2) policy to maximize Q-values and entropy,
        (3) temperature α if automatic tuning enabled, (4) soft update target networks.
        
        Args:
            state: Current algorithm state
            batch: Training batch
            rng: Random key
            
        Returns:
            Tuple of (new_state, metrics)
        """
        rng, critic_rng, actor_rng, cql_rng = jax.random.split(rng, 4)
        
        alpha = jnp.exp(state["log_alpha"])
        
        def critic_loss_fn(critic_params):
            next_actions, next_log_probs = self._sample_actions(
                state["actor"].params,
                batch.next_obs,
                critic_rng,
            )
            
            target_q = state["critic"].apply_fn(
                state["target_critic"],
                batch.next_obs,
                next_actions,
            )
            
            target_q = jnp.min(target_q, axis=-1)
            target_q = target_q - alpha * next_log_probs
            
            td_target = batch.reward + self.algo_cfg.gamma * (1 - batch.done) * target_q
            td_target = jax.lax.stop_gradient(td_target)
            
            q_values = state["critic"].apply_fn(
                critic_params,
                batch.obs,
                batch.act,
            )
            
            td_loss = jnp.mean((q_values - td_target[:, None]) ** 2)
            
            cql_penalty = self._compute_cql_penalty(
                critic_params,
                state["actor"].params,
                batch.obs,
                batch.act,
                cql_rng,
            )
            
            critic_loss = td_loss + self.algo_cfg.alpha * cql_penalty
            
            info = {
                "td_loss": td_loss,
                "cql_penalty": cql_penalty,
                "q_mean": jnp.mean(q_values),
                "target_q_mean": jnp.mean(td_target),
            }
            
            return critic_loss, info
        
        (critic_loss, critic_info), critic_grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )(state["critic"].params)
        
        critic_state = state["critic"].apply_gradients(grads=critic_grads)
        
        def actor_loss_fn(actor_params):
            actions, log_probs = self._sample_actions(
                actor_params,
                batch.obs,
                actor_rng,
            )
            
            q_values = state["critic"].apply_fn(
                critic_state.params,
                batch.obs,
                actions,
            )
            
            q_values = jnp.min(q_values, axis=-1)
            actor_loss = jnp.mean(alpha * log_probs - q_values)
            
            info = {
                "actor_q_mean": jnp.mean(q_values),
                "log_prob_mean": jnp.mean(log_probs),
                "entropy": -jnp.mean(log_probs),
            }
            
            return actor_loss, info
        
        (actor_loss, actor_info), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(state["actor"].params)
        
        actor_state = state["actor"].apply_gradients(grads=actor_grads)
        
        new_log_alpha = state["log_alpha"]
        new_alpha_opt_state = state["alpha_opt_state"]
        alpha_loss = 0.0
        
        if self.algo_cfg.use_automatic_entropy_tuning:
            def alpha_loss_fn(log_alpha):
                alpha_loss = -jnp.mean(
                    jnp.exp(log_alpha) * (actor_info["log_prob_mean"] + self.target_entropy)
                )
                return alpha_loss
            
            alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(state["log_alpha"])
            
            updates, new_alpha_opt_state = self.alpha_tx.update(
                alpha_grads, state["alpha_opt_state"]
            )
            new_log_alpha = optax.apply_updates(state["log_alpha"], updates)
        
        target_critic_params = jax.tree.map(
            lambda target, online: self.algo_cfg.tau * online + (1 - self.algo_cfg.tau) * target,
            state["target_critic"],
            critic_state.params,
        )
        
        new_state = {
            "actor": actor_state,
            "critic": critic_state,
            "target_critic": target_critic_params,
            "log_alpha": new_log_alpha,
            "alpha_opt_state": new_alpha_opt_state,
            "step": state["step"] + 1,
            "rng": rng,
        }
        
        metrics = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "td_loss": critic_info["td_loss"],
            "cql_penalty": critic_info["cql_penalty"],
            "q_mean": critic_info["q_mean"],
            "target_q_mean": critic_info["target_q_mean"],
            "actor_q_mean": actor_info["actor_q_mean"],
            "log_prob_mean": actor_info["log_prob_mean"],
            "entropy": actor_info["entropy"],
            "alpha": jnp.exp(new_log_alpha),
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
