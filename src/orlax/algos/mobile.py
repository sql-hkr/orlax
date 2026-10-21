"""MOBILE (Model-Based Offline RL with Local Uncertainty Regularization) algorithm implementation."""

from typing import Any, Dict
from functools import partial
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
import distrax

from ..core.types import Batch
from ..core.cfg import TrainingCfg, ModelCfg, AlgoCfg
from ..models.nets import Actor, Critic, EnsembleDynamics


class MOBILE:
    r"""Model-Based Offline RL with Local Uncertainty Regularization (MOBILE).
    
    MOBILE directly regularizes Bellman backups using local model uncertainty rather
    than cumulative trajectory reward penalties:
    
    $$y(s, a) = r(s, a) + \gamma \min_{i} Q'_i(s', a') - \lambda \mathcal{U}(s, a)$$
    
    where $\mathcal{U}(s, a)$ measures the ensemble disagreement at $(s, a)$.
    
    Reference: Sun et al., "Model-Based Offline Policy Optimization with Local Uncertainty Regularization" (2023)
    https://arxiv.org/abs/2303.05193
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
        
        self.lmbda = algo_cfg.lmbda
        
        self.dynamics = EnsembleDynamics(
            obs_dim=obs_dim,
            num_models=algo_cfg.n_models,
            hidden_dims=model_cfg.hidden_dims,
            activation=model_cfg.activation,
        )
        
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
        rng, dyn_rng, actor_rng, critic_rng = jax.random.split(rng, 4)
        
        dummy_obs = jnp.zeros((1, self.obs_dim))
        dummy_act = jnp.zeros((1, self.act_dim))
        
        dyn_params = self.dynamics.init(dyn_rng, dummy_obs, dummy_act)
        actor_params = self.actor.init(actor_rng, dummy_obs)
        critic_params = self.critic.init(critic_rng, dummy_obs, dummy_act)
        
        dyn_state = TrainState.create(
            apply_fn=self.dynamics.apply,
            params=dyn_params,
            tx=optax.adam(learning_rate=self.cfg.lr),
        )
        
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
            "dynamics": dyn_state,
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
        """Perform one MOBILE training step."""
        rng, roll_rng, critic_rng, actor_rng = jax.random.split(rng, 4)
        alpha = jnp.exp(state["log_alpha"])
        
        # 1. Train Dynamics Model
        target_delta = batch.next_obs - batch.obs
        def dyn_loss_fn(dyn_params):
            pred_deltas, pred_rews = self.dynamics.apply(dyn_params, batch.obs, batch.act)
            delta_loss = jnp.mean((pred_deltas - target_delta[:, None, :]) ** 2)
            rew_loss = jnp.mean((pred_rews - batch.reward[:, None]) ** 2)
            return delta_loss + rew_loss, {"dyn_loss": delta_loss + rew_loss}
        
        (dyn_loss, dyn_info), dyn_grads = jax.value_and_grad(dyn_loss_fn, has_aux=True)(state["dynamics"].params)
        dyn_state = state["dynamics"].apply_gradients(grads=dyn_grads)
        
        # 2. Local Uncertainty Estimation at (batch.obs, batch.act)
        pred_deltas, _ = self.dynamics.apply(dyn_state.params, batch.obs, batch.act)
        local_uncertainty = jnp.mean(jnp.std(pred_deltas, axis=1), axis=-1)  # [batch_size]
        
        # 3. Critic Update with Local Uncertainty Regularized Bellman Target
        def critic_loss_fn(critic_params):
            next_act, next_log_prob = self._sample_action(state["actor"].params, batch.next_obs, critic_rng)
            target_q = state["critic"].apply_fn(state["target_critic"], batch.next_obs, next_act)
            min_target_q = jnp.min(target_q, axis=-1) - alpha * next_log_prob
            
            # Local uncertainty penalized TD target
            td_target = batch.reward + self.algo_cfg.gamma * (1.0 - batch.done) * min_target_q - self.lmbda * local_uncertainty
            td_target = jax.lax.stop_gradient(td_target)
            
            q_values = state["critic"].apply_fn(critic_params, batch.obs, batch.act)
            critic_loss = jnp.mean((q_values - td_target[:, None]) ** 2)
            return critic_loss, {"q_mean": jnp.mean(q_values), "target_q_mean": jnp.mean(td_target)}
        
        (critic_loss, critic_info), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(state["critic"].params)
        critic_state = state["critic"].apply_gradients(grads=critic_grads)
        
        # 4. Policy Update
        def actor_loss_fn(actor_params):
            actions, log_probs = self._sample_action(actor_params, batch.obs, actor_rng)
            q_vals = state["critic"].apply_fn(critic_state.params, batch.obs, actions)
            min_q = jnp.min(q_vals, axis=-1)
            actor_loss = jnp.mean(alpha * log_probs - min_q)
            return actor_loss, {"actor_q": jnp.mean(min_q), "entropy": -jnp.mean(log_probs)}
        
        (actor_loss, actor_info), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(state["actor"].params)
        actor_state = state["actor"].apply_gradients(grads=actor_grads)
        
        # 5. Temperature update
        new_log_alpha = state["log_alpha"]
        new_alpha_opt_state = state["alpha_opt_state"]
        if self.algo_cfg.use_automatic_entropy_tuning:
            def alpha_loss_fn(log_alpha):
                return -jnp.mean(jnp.exp(log_alpha) * (-actor_info["entropy"] + self.target_entropy))
            alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(state["log_alpha"])
            updates, new_alpha_opt_state = self.alpha_tx.update(alpha_grads, state["alpha_opt_state"])
            new_log_alpha = optax.apply_updates(state["log_alpha"], updates)
        
        # 6. Target Polyak Averaging
        target_critic = jax.tree.map(
            lambda t, o: self.algo_cfg.tau * o + (1.0 - self.algo_cfg.tau) * t,
            state["target_critic"],
            critic_state.params,
        )
        
        new_state = {
            "dynamics": dyn_state,
            "actor": actor_state,
            "critic": critic_state,
            "target_critic": target_critic,
            "log_alpha": new_log_alpha,
            "alpha_opt_state": new_alpha_opt_state,
            "step": state["step"] + 1,
            "rng": rng,
        }
        
        metrics = {
            "dyn_loss": dyn_loss,
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "local_uncertainty": jnp.mean(local_uncertainty),
            "q_mean": critic_info["q_mean"],
        }
        return new_state, metrics
    
    def eval_step(self, state: Dict[str, Any], obs: jnp.ndarray) -> jnp.ndarray:
        mean, _ = state["actor"].apply_fn(state["actor"].params, obs)
        return mean
