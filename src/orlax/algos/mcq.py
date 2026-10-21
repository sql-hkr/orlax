"""Masked Conservative Q-Learning (MCQ) algorithm implementation."""

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


class MCQ:
    r"""Masked Conservative Q-Learning (MCQ) for offline RL.
    
    MCQ avoids the excessive conservatism and heavy sampling of standard CQL by
    sampling actions from the current policy and masking out in-distribution actions.
    It penalizes Q-values only on truly OOD actions:
    
    $$\mathcal{L}_{MCQ}(Q) = \mathbb{E}_{(s, a) \sim \mathcal{D}, a_{ood} \sim \pi} \left[ M(s, a_{ood}) \cdot \max(0, Q(s, a_{ood}) - Q(s, a))^2 \right]$$
    
    Reference: Lyu et al., "Mildly Conservative Q-Learning for Offline Reinforcement Learning" (NeurIPS 2022)
    https://arxiv.org/abs/2206.04745
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
        
        self.eta = algo_cfg.eta  # MCQ conservative weight
        self.delta = 0.3  # Action distance threshold for OOD masking
        
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
        
        self.target_entropy = -float(act_dim)
    
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
    
    def _sample_action(self, actor_params: Any, obs: jnp.ndarray, rng: jnp.ndarray):
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
        """Perform one MCQ training step."""
        rng, critic_rng, ood_rng, actor_rng = jax.random.split(rng, 4)
        alpha = jnp.exp(state["log_alpha"])
        
        # 1. Critic Update with Masked Conservative Regularization
        def critic_loss_fn(critic_params):
            next_act, next_log_prob = self._sample_action(state["actor"].params, batch.next_obs, critic_rng)
            target_q = state["critic"].apply_fn(state["target_critic"], batch.next_obs, next_act)
            min_target_q = jnp.min(target_q, axis=-1) - alpha * next_log_prob
            
            td_target = batch.reward + self.algo_cfg.gamma * (1.0 - batch.done) * min_target_q
            td_target = jax.lax.stop_gradient(td_target)
            
            q_data = state["critic"].apply_fn(critic_params, batch.obs, batch.act)
            td_loss = jnp.mean((q_data - td_target[:, None]) ** 2)
            
            # Sample OOD actions from current policy
            ood_actions, _ = self._sample_action(state["actor"].params, batch.obs, ood_rng)
            
            # Compute distance between OOD action and dataset action
            action_dist = jnp.linalg.norm(ood_actions - batch.act, axis=-1)
            # Soft mask: 1 if OOD (dist > delta), 0 if near data
            mask = jax.nn.sigmoid((action_dist - self.delta) / 0.1)
            mask = jax.lax.stop_gradient(mask)
            
            q_ood = state["critic"].apply_fn(critic_params, batch.obs, ood_actions)
            
            # Penalize Q_ood exceeding Q_data only where masked as OOD
            q_diff = jnp.maximum(0.0, q_ood - jax.lax.stop_gradient(q_data))
            mcq_loss = jnp.mean(mask[:, None] * (q_diff ** 2))
            
            total_critic_loss = td_loss + self.eta * mcq_loss
            return total_critic_loss, {
                "td_loss": td_loss,
                "mcq_loss": mcq_loss,
                "q_mean": jnp.mean(q_data),
                "mask_mean": jnp.mean(mask),
            }
        
        (critic_loss, critic_info), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(state["critic"].params)
        critic_state = state["critic"].apply_gradients(grads=critic_grads)
        
        # 2. Policy Update
        def actor_loss_fn(actor_params):
            actions, log_probs = self._sample_action(actor_params, batch.obs, actor_rng)
            q_vals = state["critic"].apply_fn(critic_state.params, batch.obs, actions)
            min_q = jnp.min(q_vals, axis=-1)
            actor_loss = jnp.mean(alpha * log_probs - min_q)
            return actor_loss, {"actor_q": jnp.mean(min_q), "entropy": -jnp.mean(log_probs)}
        
        (actor_loss, actor_info), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(state["actor"].params)
        actor_state = state["actor"].apply_gradients(grads=actor_grads)
        
        # 3. Temperature update
        new_log_alpha = state["log_alpha"]
        new_alpha_opt_state = state["alpha_opt_state"]
        if self.algo_cfg.use_automatic_entropy_tuning:
            def alpha_loss_fn(log_alpha):
                return -jnp.mean(jnp.exp(log_alpha) * (-actor_info["entropy"] + self.target_entropy))
            alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(state["log_alpha"])
            updates, new_alpha_opt_state = self.alpha_tx.update(alpha_grads, state["alpha_opt_state"])
            new_log_alpha = optax.apply_updates(state["log_alpha"], updates)
        
        # 4. Target Polyak Averaging
        target_critic = jax.tree.map(
            lambda t, o: self.algo_cfg.tau * o + (1.0 - self.algo_cfg.tau) * t,
            state["target_critic"],
            critic_state.params,
        )
        
        new_state = {
            "actor": actor_state,
            "critic": critic_state,
            "target_critic": target_critic,
            "log_alpha": new_log_alpha,
            "alpha_opt_state": new_alpha_opt_state,
            "step": state["step"] + 1,
            "rng": rng,
        }
        
        metrics = {
            "critic_loss": critic_loss,
            "td_loss": critic_info["td_loss"],
            "mcq_loss": critic_info["mcq_loss"],
            "actor_loss": actor_loss,
            "mask_mean": critic_info["mask_mean"],
            "q_mean": critic_info["q_mean"],
        }
        return new_state, metrics
    
    def eval_step(self, state: Dict[str, Any], obs: jnp.ndarray) -> jnp.ndarray:
        mean, _ = state["actor"].apply_fn(state["actor"].params, obs)
        return mean
