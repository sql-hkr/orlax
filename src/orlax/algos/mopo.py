"""Model-based Offline Policy Optimization (MOPO) algorithm implementation."""

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


class MOPO:
    r"""Model-based Offline Policy Optimization (MOPO).
    
    MOPO trains an ensemble dynamics model and constructs a pessimistic MDP by
    penalizing rewards proportional to the model transition uncertainty:
    
    $$\tilde{r}(s, a) = \hat{r}(s, a) - \lambda u(s, a)$$
    
    where $u(s, a) = \max_{i} \|\hat{T}_i(s, a) - \bar{T}(s, a)\|$ captures epistemic uncertainty.
    
    Reference: Yu et al., "MOPO: Model-based Offline Policy Optimization" (NeurIPS 2020)
    https://arxiv.org/abs/2005.13239
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
        """Perform one MOPO training step."""
        rng, roll_rng, critic_rng, actor_rng = jax.random.split(rng, 4)
        alpha = jnp.exp(state["log_alpha"])
        
        # 1. Dynamics Model Update
        target_delta = batch.next_obs - batch.obs
        def dyn_loss_fn(dyn_params):
            pred_deltas, pred_rews = self.dynamics.apply(dyn_params, batch.obs, batch.act)
            delta_loss = jnp.mean((pred_deltas - target_delta[:, None, :]) ** 2)
            rew_loss = jnp.mean((pred_rews - batch.reward[:, None]) ** 2)
            return delta_loss + rew_loss, {"dyn_delta_loss": delta_loss, "dyn_rew_loss": rew_loss}
        
        (dyn_loss, dyn_info), dyn_grads = jax.value_and_grad(dyn_loss_fn, has_aux=True)(state["dynamics"].params)
        dyn_state = state["dynamics"].apply_gradients(grads=dyn_grads)
        
        # 2. Imaginary Rollout with Uncertainty Penalty
        sim_actions, _ = self._sample_action(state["actor"].params, batch.obs, roll_rng)
        pred_deltas, pred_rews = self.dynamics.apply(dyn_state.params, batch.obs, sim_actions)
        
        # Mean prediction across ensemble
        mean_delta = jnp.mean(pred_deltas, axis=1)
        mean_rew = jnp.mean(pred_rews, axis=1)
        sim_next_obs = batch.obs + mean_delta
        
        # Epistemic uncertainty: std across ensemble models
        uncertainty = jnp.mean(jnp.std(pred_deltas, axis=1), axis=-1)
        penalized_rew = mean_rew - self.lmbda * uncertainty
        
        # 3. Critic Update on both real & imaginary transitions
        comb_obs = jnp.concatenate([batch.obs, batch.obs], axis=0)
        comb_act = jnp.concatenate([batch.act, sim_actions], axis=0)
        comb_next_obs = jnp.concatenate([batch.next_obs, sim_next_obs], axis=0)
        comb_rew = jnp.concatenate([batch.reward, penalized_rew], axis=0)
        comb_done = jnp.concatenate([batch.done, batch.done], axis=0)
        
        def critic_loss_fn(critic_params):
            next_act, next_log_prob = self._sample_action(state["actor"].params, comb_next_obs, critic_rng)
            target_q = state["critic"].apply_fn(state["target_critic"], comb_next_obs, next_act)
            min_target_q = jnp.min(target_q, axis=-1) - alpha * next_log_prob
            
            td_target = comb_rew + self.algo_cfg.gamma * (1.0 - comb_done) * min_target_q
            td_target = jax.lax.stop_gradient(td_target)
            
            q_values = state["critic"].apply_fn(critic_params, comb_obs, comb_act)
            critic_loss = jnp.mean((q_values - td_target[:, None]) ** 2)
            return critic_loss, {"q_mean": jnp.mean(q_values)}
        
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
        
        # 5. Alpha Temperature Update
        new_log_alpha = state["log_alpha"]
        new_alpha_opt_state = state["alpha_opt_state"]
        if self.algo_cfg.use_automatic_entropy_tuning:
            def alpha_loss_fn(log_alpha):
                return -jnp.mean(jnp.exp(log_alpha) * (-actor_info["entropy"] + self.target_entropy))
            alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(state["log_alpha"])
            updates, new_alpha_opt_state = self.alpha_tx.update(alpha_grads, state["alpha_opt_state"])
            new_log_alpha = optax.apply_updates(state["log_alpha"], updates)
        
        # 6. Target Critic Polyak Averaging
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
            "uncertainty_mean": jnp.mean(uncertainty),
            "q_mean": critic_info["q_mean"],
        }
        return new_state, metrics
    
    def eval_step(self, state: Dict[str, Any], obs: jnp.ndarray) -> jnp.ndarray:
        mean, _ = state["actor"].apply_fn(state["actor"].params, obs)
        return mean
