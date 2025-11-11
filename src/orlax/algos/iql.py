"""Implicit Q-Learning (IQL) algorithm implementation."""

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


class IQL:
    """Implicit Q-Learning for offline RL.
    
    IQL avoids distributional shift through a two-stage approach:
    1. Learn Q-functions and value function V via expectile regression without querying the policy
    2. Extract policy via advantage-weighted behavioral cloning
    
    The value function approximates $\max_a Q(s,a)$ via expectile loss:
    
    $$L_V(\tau) = \mathbb{E}[L_2^\tau(Q(s,a) - V(s))]$$
    
    where $L_2^\tau$ is the asymmetric expectile loss with $\tau \in [0.7, 0.9]$.
    
    Reference: Kostrikov et al., "Offline RL with Implicit Q-Learning" (2021)
    https://arxiv.org/abs/2110.06169
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
        
        self.value = ValueNetwork(
            hidden_dims=model_cfg.hidden_dims,
            activation=model_cfg.activation,
        )
    
    def init(self, rng: jnp.ndarray) -> Dict[str, Any]:
        """Initialize algorithm state.
        
        Args:
            rng: JAX random key
            
        Returns:
            Dictionary containing actor, critic, value, target critic, step counter, and RNG key.
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
        
        target_critic_params = jax.tree.map(lambda x: x, critic_params)
        
        return {
            "actor": actor_state,
            "critic": critic_state,
            "value": value_state,
            "target_critic": target_critic_params,
            "step": 0,
            "rng": rng,
        }
    
    def _expectile_loss(
        self,
        diff: jnp.ndarray,
        expectile: float = 0.7,
    ) -> jnp.ndarray:
        """Compute asymmetric expectile loss.
        
        $$L_2^\tau(\delta) = |\tau - \mathbb{1}(\delta < 0)| \cdot \delta^2$$
        
        For $\tau > 0.5$, penalizes positive errors more, pushing V towards upper quantiles of Q.
        
        Args:
            diff: Error (target - prediction) [batch_size]
            expectile: Expectile parameter $\tau \in [0, 1]$
            
        Returns:
            Expectile loss [batch_size]
        """
        weight = jnp.where(diff > 0, expectile, 1 - expectile)
        return weight * (diff ** 2)
    
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
    
    @partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: Dict[str, Any],
        batch: Batch,
        rng: jnp.ndarray,
    ) -> tuple[Dict[str, Any], Dict[str, float]]:
        """Perform one IQL training step.
        
        Updates: (1) Value V via expectile regression on Q-V, (2) Q-functions via TD learning using V(s'),
        (3) Policy via advantage-weighted behavioral cloning, (4) soft update target networks.
        
        Args:
            state: Current algorithm state
            batch: Training batch
            rng: Random key
            
        Returns:
            Tuple of (new_state, metrics)
        """
        rng, actor_rng = jax.random.split(rng)
        
        def value_loss_fn(value_params):
            q_values = state["critic"].apply_fn(
                state["target_critic"],
                batch.obs,
                batch.act,
            )
            
            q_values = jnp.min(q_values, axis=-1)
            v_values = state["value"].apply_fn(value_params, batch.obs)
            
            diff = q_values - v_values
            expectile_loss = self._expectile_loss(diff, self.algo_cfg.expectile)
            value_loss = jnp.mean(expectile_loss)
            
            info = {
                "v_mean": jnp.mean(v_values),
                "v_std": jnp.std(v_values),
                "advantage_mean": jnp.mean(diff),
                "advantage_std": jnp.std(diff),
            }
            
            return value_loss, info
        
        (value_loss, value_info), value_grads = jax.value_and_grad(
            value_loss_fn, has_aux=True
        )(state["value"].params)
        
        value_state = state["value"].apply_gradients(grads=value_grads)
        
        def critic_loss_fn(critic_params):
            target_v = state["value"].apply_fn(
                value_state.params,
                batch.next_obs,
            )
            
            td_target = batch.reward + self.algo_cfg.gamma * (1 - batch.done) * target_v
            td_target = jax.lax.stop_gradient(td_target)
            
            q_values = state["critic"].apply_fn(
                critic_params,
                batch.obs,
                batch.act,
            )
            
            td_loss = jnp.mean((q_values - td_target[:, None]) ** 2)
            
            info = {
                "q_mean": jnp.mean(q_values),
                "q_std": jnp.std(q_values),
                "target_v_mean": jnp.mean(target_v),
                "td_target_mean": jnp.mean(td_target),
            }
            
            return td_loss, info
        
        (critic_loss, critic_info), critic_grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )(state["critic"].params)
        
        critic_state = state["critic"].apply_gradients(grads=critic_grads)
        
        def actor_loss_fn(actor_params):
            """Advantage-weighted behavioral cloning: $L = \mathbb{E}[\exp(\beta A(s,a)) \cdot ||a - \pi(s)||^2]$"""
            q_values = state["critic"].apply_fn(
                critic_state.params,
                batch.obs,
                batch.act,
            )
            
            q_values = jnp.min(q_values, axis=-1)
            
            v_values = state["value"].apply_fn(
                value_state.params,
                batch.obs,
            )
            
            advantages = q_values - v_values
            weights = jnp.exp(self.algo_cfg.beta * advantages)
            weights = jnp.minimum(weights, 100.0)
            
            pred_actions, _ = state["actor"].apply_fn(actor_params, batch.obs)
            squared_errors = jnp.sum((pred_actions - batch.act) ** 2, axis=-1)
            actor_loss = jnp.mean(weights * squared_errors)
            
            info = {
                "actor_loss_unweighted": jnp.mean(squared_errors),
                "advantage_weights_mean": jnp.mean(weights),
                "advantage_weights_max": jnp.max(weights),
                "action_mean": jnp.mean(jnp.abs(pred_actions)),
            }
            
            return actor_loss, info
        
        (actor_loss, actor_info), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(state["actor"].params)
        
        actor_state = state["actor"].apply_gradients(grads=actor_grads)
        
        target_critic_params = jax.tree.map(
            lambda target, online: self.algo_cfg.tau * online + (1 - self.algo_cfg.tau) * target,
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
            "advantage_mean": value_info["advantage_mean"],
            "advantage_std": value_info["advantage_std"],
            "q_mean": critic_info["q_mean"],
            "q_std": critic_info["q_std"],
            "target_v_mean": critic_info["target_v_mean"],
            "td_target_mean": critic_info["td_target_mean"],
            "actor_loss_unweighted": actor_info["actor_loss_unweighted"],
            "advantage_weights_mean": actor_info["advantage_weights_mean"],
            "advantage_weights_max": actor_info["advantage_weights_max"],
            "action_mean": actor_info["action_mean"],
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
