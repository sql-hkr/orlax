"""Critic Regularized Regression (CRR) algorithm implementation."""

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


class CRR:
    r"""Critic Regularized Regression (CRR) for offline RL.
    
    CRR updates the policy via advantage-weighted behavioral cloning:
    
    $$\max_\theta \mathbb{E}_{(s, a) \sim \mathcal{D}} [f(Q, \pi, s, a) \log \pi_\theta(a|s)]$$
    
    where $f(Q, \pi, s, a)$ is a non-negative weight function of the estimated advantage:
    $A(s, a) = Q(s, a) - \mathbb{E}_{a' \sim \pi}[Q(s, a')]$.
    
    Reference: Wang et al., "Critic Regularized Regression" (NeurIPS 2020)
    https://arxiv.org/abs/2006.15134
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
    
    def init(self, rng: jnp.ndarray) -> Dict[str, Any]:
        """Initialize algorithm state."""
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        
        dummy_obs = jnp.zeros((1, self.obs_dim))
        dummy_act = jnp.zeros((1, self.act_dim))
        
        actor_params = self.actor.init(actor_rng, dummy_obs)
        critic_params = self.critic.init(critic_rng, dummy_obs, dummy_act)
        
        actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=self.cfg.lr),
        )
        
        critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=self.cfg.lr),
        )
        
        target_critic_params = jax.tree.map(lambda x: x, critic_params)
        
        return {
            "actor": actor_state,
            "critic": critic_state,
            "target_critic": target_critic_params,
            "step": 0,
            "rng": rng,
        }
    
    def _sample_action(self, actor_params: Any, obs: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        mean, log_std = self.actor.apply(actor_params, obs)
        std = jnp.exp(log_std)
        dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
        return dist.sample(seed=rng)
    
    @partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: Dict[str, Any],
        batch: Batch,
        rng: jnp.ndarray,
    ) -> tuple[Dict[str, Any], Dict[str, float]]:
        """Perform one CRR training step."""
        rng, critic_rng, actor_sample_rng = jax.random.split(rng, 3)
        
        # 1. Critic Update (standard TD learning)
        def critic_loss_fn(critic_params):
            next_actions = self._sample_action(state["actor"].params, batch.next_obs, critic_rng)
            target_q = state["critic"].apply_fn(state["target_critic"], batch.next_obs, next_actions)
            target_q = jnp.min(target_q, axis=-1)
            
            td_target = batch.reward + self.algo_cfg.gamma * (1.0 - batch.done) * target_q
            td_target = jax.lax.stop_gradient(td_target)
            
            q_values = state["critic"].apply_fn(critic_params, batch.obs, batch.act)
            critic_loss = jnp.mean((q_values - td_target[:, None]) ** 2)
            
            return critic_loss, {"q_mean": jnp.mean(q_values), "target_q_mean": jnp.mean(td_target)}
        
        (critic_loss, critic_info), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(state["critic"].params)
        critic_state = state["critic"].apply_gradients(grads=critic_grads)
        
        # 2. Advantage-weighted Policy Update
        def actor_loss_fn(actor_params):
            # Q(s, a_dataset)
            q_data = state["critic"].apply_fn(critic_state.params, batch.obs, batch.act)
            q_data = jnp.min(q_data, axis=-1)
            
            # Q(s, a_policy) for advantage baseline
            policy_actions = self._sample_action(actor_params, batch.obs, actor_sample_rng)
            q_policy = state["critic"].apply_fn(critic_state.params, batch.obs, policy_actions)
            q_policy = jnp.min(q_policy, axis=-1)
            
            advantage = q_data - q_policy
            
            # Compute weights f(A)
            beta = self.algo_cfg.beta
            if self.algo_cfg.crr_weight_type == "binary":
                weights = (advantage > 0.0).astype(jnp.float32)
            else:  # "exp"
                weights = jnp.clip(jnp.exp(advantage / beta), 0.0, 20.0)
            weights = jax.lax.stop_gradient(weights)
            
            mean, log_std = state["actor"].apply_fn(actor_params, batch.obs)
            dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
            log_probs = dist.log_prob(batch.act)
            
            actor_loss = -jnp.mean(weights * log_probs)
            
            info = {
                "actor_loss": actor_loss,
                "advantage_mean": jnp.mean(advantage),
                "weights_mean": jnp.mean(weights),
            }
            return actor_loss, info
        
        (actor_loss, actor_info), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(state["actor"].params)
        actor_state = state["actor"].apply_gradients(grads=actor_grads)
        
        # 3. Soft target update
        target_critic = jax.tree.map(
            lambda t, o: self.algo_cfg.tau * o + (1.0 - self.algo_cfg.tau) * t,
            state["target_critic"],
            critic_state.params,
        )
        
        new_state = {
            "actor": actor_state,
            "critic": critic_state,
            "target_critic": target_critic,
            "step": state["step"] + 1,
            "rng": rng,
        }
        
        metrics = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "q_mean": critic_info["q_mean"],
            "advantage_mean": actor_info["advantage_mean"],
            "weights_mean": actor_info["weights_mean"],
        }
        return new_state, metrics
    
    def eval_step(self, state: Dict[str, Any], obs: jnp.ndarray) -> jnp.ndarray:
        """Deterministic evaluation action."""
        mean, _ = state["actor"].apply_fn(state["actor"].params, obs)
        return mean
