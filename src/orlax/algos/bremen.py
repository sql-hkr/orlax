"""BREMEN (Behavior-Regularized Model-Ensemble) algorithm implementation."""

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


class BREMEN:
    r"""Behavior-Regularized Model-Ensemble (BREMEN).
    
    BREMEN combines model-based imaginary rollouts from a dynamics ensemble with
    behavior regularization to prevent model exploitation:
    
    $$\max_\theta \mathbb{E}_{s \sim d_{model}} [Q(s, \pi_\theta(s))] - \lambda \mathbb{E}_{(s, a) \sim \mathcal{D}} [(\pi_\theta(s) - a)^2]$$
    
    Reference: Matsushima et al., "Deployment-Efficient Reinforcement Learning via Model-Based Offline Pre-Training and On-Policy Fine-Tuning" (2020)
    https://arxiv.org/abs/2006.03669
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
        
        self.bc_weight = algo_cfg.lmbda  # Behavior regularization weight
        
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
        
        return {
            "dynamics": dyn_state,
            "actor": actor_state,
            "critic": critic_state,
            "target_critic": target_critic_params,
            "step": 0,
            "rng": rng,
        }
    
    def _sample_action(self, actor_params: Any, obs: jnp.ndarray, rng: jnp.ndarray):
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
        """Perform one BREMEN training step."""
        rng, roll_rng, critic_rng, actor_rng = jax.random.split(rng, 4)
        
        # 1. Dynamics Model Training (MSE loss)
        target_delta = batch.next_obs - batch.obs
        def dyn_loss_fn(dyn_params):
            pred_deltas, pred_rews = self.dynamics.apply(dyn_params, batch.obs, batch.act)
            delta_loss = jnp.mean((pred_deltas - target_delta[:, None, :]) ** 2)
            rew_loss = jnp.mean((pred_rews - batch.reward[:, None]) ** 2)
            return delta_loss + rew_loss, {"dyn_loss": delta_loss + rew_loss}
        
        (dyn_loss, dyn_info), dyn_grads = jax.value_and_grad(dyn_loss_fn, has_aux=True)(state["dynamics"].params)
        dyn_state = state["dynamics"].apply_gradients(grads=dyn_grads)
        
        # 2. Imaginary Rollout from Ensemble
        sim_actions = self._sample_action(state["actor"].params, batch.obs, roll_rng)
        pred_deltas, pred_rews = self.dynamics.apply(dyn_state.params, batch.obs, sim_actions)
        sim_next_obs = batch.obs + jnp.mean(pred_deltas, axis=1)
        sim_rew = jnp.mean(pred_rews, axis=1)
        
        # Combine real & imaginary transitions
        comb_obs = jnp.concatenate([batch.obs, batch.obs], axis=0)
        comb_act = jnp.concatenate([batch.act, sim_actions], axis=0)
        comb_next_obs = jnp.concatenate([batch.next_obs, sim_next_obs], axis=0)
        comb_rew = jnp.concatenate([batch.reward, sim_rew], axis=0)
        comb_done = jnp.concatenate([batch.done, batch.done], axis=0)
        
        # 3. Critic Update
        def critic_loss_fn(critic_params):
            next_act = self._sample_action(state["actor"].params, comb_next_obs, critic_rng)
            target_q = state["critic"].apply_fn(state["target_critic"], comb_next_obs, next_act)
            min_target_q = jnp.min(target_q, axis=-1)
            
            td_target = comb_rew + self.algo_cfg.gamma * (1.0 - comb_done) * min_target_q
            td_target = jax.lax.stop_gradient(td_target)
            
            q_values = state["critic"].apply_fn(critic_params, comb_obs, comb_act)
            critic_loss = jnp.mean((q_values - td_target[:, None]) ** 2)
            return critic_loss, {"q_mean": jnp.mean(q_values)}
        
        (critic_loss, critic_info), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(state["critic"].params)
        critic_state = state["critic"].apply_gradients(grads=critic_grads)
        
        # 4. Policy Update with Behavior Regularization
        def actor_loss_fn(actor_params):
            pred_actions = self._sample_action(actor_params, batch.obs, actor_rng)
            q_vals = state["critic"].apply_fn(critic_state.params, batch.obs, pred_actions)
            q_mean = jnp.mean(jnp.min(q_vals, axis=-1))
            
            # Behavioral cloning regularizer on real dataset actions
            det_actions, _ = state["actor"].apply_fn(actor_params, batch.obs)
            bc_loss = jnp.mean((det_actions - batch.act) ** 2)
            
            actor_loss = -q_mean + self.bc_weight * bc_loss
            return actor_loss, {"actor_q": q_mean, "bc_loss": bc_loss}
        
        (actor_loss, actor_info), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(state["actor"].params)
        actor_state = state["actor"].apply_gradients(grads=actor_grads)
        
        # 5. Target Polyak Averaging
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
            "step": state["step"] + 1,
            "rng": rng,
        }
        
        metrics = {
            "dyn_loss": dyn_loss,
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "bc_loss": actor_info["bc_loss"],
            "q_mean": critic_info["q_mean"],
        }
        return new_state, metrics
    
    def eval_step(self, state: Dict[str, Any], obs: jnp.ndarray) -> jnp.ndarray:
        mean, _ = state["actor"].apply_fn(state["actor"].params, obs)
        return mean
