"""Extreme Q-Learning (XQL) algorithm implementation."""

from typing import Any, Dict
from functools import partial
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
import distrax

from ..core.types import Batch
from ..core.cfg import TrainingCfg, ModelCfg, AlgoCfg
from ..models.nets import Actor, Critic, ValueNetwork


class XQL:
    """Extreme Q-Learning for offline RL.
    
    XQL computes the optimal state-value function V*(s) = max_a Q(s, a) without
    querying out-of-distribution actions. It uses Extreme Value Theory (EVT) and Gumbel
    regression to estimate the maximum Q-value from offline dataset actions, and
    extracts the policy via advantage-weighted regression.
    
    Reference: Garg et al., "Extreme Q-Learning: Infeasible or Inevitable?"
    (ICLR 2023 / arXiv:2303.01486)
    https://arxiv.org/abs/2303.01486
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
        
        self.critic = Critic(
            hidden_dims=model_cfg.hidden_dims,
            activation=model_cfg.activation,
            num_critics=algo_cfg.n_critics,
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
            Dictionary containing actor, critic, value, target critic, step counter, and RNG.
        """
        rng, actor_rng, critic_rng, value_rng = jax.random.split(rng, 4)
        
        dummy_obs = jnp.zeros((1, self.obs_dim))
        dummy_act = jnp.zeros((1, self.act_dim))
        
        actor_params = self.actor.init(actor_rng, dummy_obs)
        critic_params = self.critic.init(critic_rng, dummy_obs, dummy_act)
        value_params = self.value.init(value_rng, dummy_obs)
        
        actor_tx = optax.adam(learning_rate=self.cfg.lr)
        critic_tx = optax.adam(learning_rate=self.cfg.lr)
        value_tx = optax.adam(learning_rate=self.cfg.lr)
        
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
        
        value_state = TrainState.create(
            apply_fn=self.value.apply,
            params=value_params,
            tx=value_tx,
        )
        
        return {
            "actor": actor_state,
            "critic": critic_state,
            "value": value_state,
            "target_critic": critic_params,
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
        """Perform one XQL training step."""
        rng, _ = jax.random.split(rng)
        
        # 1. Update Value function V via Gumbel regression
        target_q = state["critic"].apply_fn(
            state["target_critic"],
            batch.obs,
            batch.act,
        )
        min_target_q = jnp.min(target_q, axis=-1)
        
        def value_loss_fn(value_params):
            v_pred = state["value"].apply_fn(value_params, batch.obs)
            delta = (min_target_q - v_pred) / self.algo_cfg.beta
            delta_clipped = jnp.minimum(delta, self.algo_cfg.xql_clip_max)
            gumbel_loss = jnp.exp(delta_clipped) - delta_clipped - 1.0
            value_loss = jnp.mean(gumbel_loss)
            info = {
                "v_mean": jnp.mean(v_pred),
                "v_std": jnp.std(v_pred),
                "delta_mean": jnp.mean(delta),
                "delta_max": jnp.max(delta),
            }
            return value_loss, info

        (value_loss, value_info), value_grads = jax.value_and_grad(
            value_loss_fn, has_aux=True
        )(state["value"].params)
        value_state = state["value"].apply_gradients(grads=value_grads)
        
        # 2. Update Critic networks Q via TD learning with target V
        target_v = state["value"].apply_fn(value_state.params, batch.next_obs)
        td_target = batch.reward + self.algo_cfg.gamma * (1.0 - batch.done) * target_v
        td_target = jax.lax.stop_gradient(td_target)
        
        def critic_loss_fn(critic_params):
            current_q = state["critic"].apply_fn(
                critic_params,
                batch.obs,
                batch.act,
            )
            td_loss = jnp.mean((current_q - td_target[:, None]) ** 2)
            info = {
                "q_mean": jnp.mean(current_q),
                "q_std": jnp.std(current_q),
                "td_target_mean": jnp.mean(td_target),
            }
            return td_loss, info

        (critic_loss, critic_info), critic_grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )(state["critic"].params)
        critic_state = state["critic"].apply_gradients(grads=critic_grads)
        
        # 3. Update Actor via Advantage-Weighted Regression
        current_v = state["value"].apply_fn(value_state.params, batch.obs)
        adv = (min_target_q - current_v) / self.algo_cfg.beta
        adv_clipped = jnp.minimum(adv, self.algo_cfg.xql_clip_max)
        weights = jnp.exp(adv_clipped)
        weights = jnp.minimum(weights, 100.0)
        weights = jax.lax.stop_gradient(weights)
        
        def actor_loss_fn(actor_params):
            mean, log_std = state["actor"].apply_fn(actor_params, batch.obs)
            std = jnp.exp(log_std)
            dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
            log_probs = dist.log_prob(batch.act)
            actor_loss = - jnp.mean(weights * log_probs)
            info = {
                "actor_loss": actor_loss,
                "log_prob_mean": jnp.mean(log_probs),
                "weights_mean": jnp.mean(weights),
            }
            return actor_loss, info

        (actor_loss, actor_info), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(state["actor"].params)
        actor_state = state["actor"].apply_gradients(grads=actor_grads)
        
        # 4. Target critic soft update
        target_critic_params = jax.tree.map(
            lambda target, online: self.algo_cfg.tau * online + (1.0 - self.algo_cfg.tau) * target,
            state["target_critic"],
            critic_state.params,
        )
        
        new_state = {
            "actor": actor_state,
            "critic": critic_state,
            "value": value_state,
            "target_critic": target_critic_params,
            "step": state["step"] + 1,
            "rng": rng,
        }
        
        metrics = {
            "value_loss": value_loss,
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "v_mean": value_info["v_mean"],
            "v_std": value_info["v_std"],
            "delta_mean": value_info["delta_mean"],
            "delta_max": value_info["delta_max"],
            "q_mean": critic_info["q_mean"],
            "q_std": critic_info["q_std"],
            "td_target_mean": critic_info["td_target_mean"],
            "weights_mean": actor_info["weights_mean"],
        }
        
        return new_state, metrics

    def eval_step(self, state: Dict[str, Any], obs: jnp.ndarray) -> jnp.ndarray:
        """Deterministic evaluation step returning mean action.
        
        Args:
            state: Algorithm state
            obs: Observation [obs_dim] or [batch_size, obs_dim]
            
        Returns:
            Action [act_dim] or [batch_size, act_dim]
        """
        mean, _ = state["actor"].apply_fn(state["actor"].params, obs)
        return mean
