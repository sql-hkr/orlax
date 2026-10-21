"""Efficient Diffusion Policy (EDP) algorithm implementation."""

from typing import Any, Dict
from functools import partial
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from ..core.types import Batch
from ..core.cfg import TrainingCfg, ModelCfg, AlgoCfg
from ..models.nets import Critic, ValueNetwork, DiffusionModel, DiffusionSchedule


class EDP:
    """Efficient Diffusion Policy (EDP) for offline RL.
    
    EDP accelerates diffusion policy training by decoupling critic learning from policy
    rollouts. Value and critic networks are trained without querying the diffusion
    policy, while the diffusion policy is trained via advantage-weighted score matching
    directly on offline transitions.
    
    Reference: Kang et al., "Efficient Diffusion Policies for Offline Reinforcement
    Learning" (NeurIPS 2023 / arXiv:2305.20081)
    https://arxiv.org/abs/2305.20081
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
        self.n_timesteps = algo_cfg.num_diffusion_steps
        
        self.actor = DiffusionModel(
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
        
        self.schedule = DiffusionSchedule.make(n_steps=self.n_timesteps)

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
        dummy_t = jnp.zeros((1,), dtype=jnp.float32)
        
        actor_params = self.actor.init(actor_rng, dummy_obs, dummy_act, dummy_t)
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

    def _sample_actions(
        self,
        actor_params: Any,
        obs: jnp.ndarray,
        rng: jnp.ndarray,
        deterministic: bool = False,
    ) -> jnp.ndarray:
        """Sample actions from diffusion policy via reverse DDPM denoising chain."""
        batch_size = obs.shape[0]
        rng, init_rng = jax.random.split(rng)
        
        if deterministic:
            x_init = jnp.zeros((batch_size, self.act_dim))
        else:
            x_init = jax.random.normal(init_rng, (batch_size, self.act_dim))
            
        def step_fn(carry, t_idx):
            x, step_rng = carry
            step_rng, noise_rng = jax.random.split(step_rng)
            z = jax.random.normal(noise_rng, x.shape)
            
            t_arr = jnp.full((batch_size,), t_idx, dtype=jnp.float32)
            eps = self.actor.apply(actor_params, obs, x, t_arr)
            
            alpha_t = self.schedule.alphas[t_idx]
            beta_t = self.schedule.betas[t_idx]
            sqrt_one_minus_alpha_bar = self.schedule.sqrt_one_minus_alphas_cumprod[t_idx]
            sigma = jnp.sqrt(self.schedule.posterior_variance[t_idx])
            
            mu = (1.0 / jnp.sqrt(alpha_t)) * (x - (beta_t / sqrt_one_minus_alpha_bar) * eps)
            
            if deterministic:
                next_x = mu
            else:
                next_x = jnp.where(t_idx == 0, mu, mu + sigma * z)
                
            return (next_x, step_rng), None

        timesteps = jnp.arange(self.n_timesteps - 1, -1, -1)
        (final_x, _), _ = jax.lax.scan(step_fn, (x_init, rng), timesteps)
        return jnp.clip(final_x, -1.0, 1.0)

    @partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: Dict[str, Any],
        batch: Batch,
        rng: jnp.ndarray,
    ) -> tuple[Dict[str, Any], Dict[str, float]]:
        """Perform one EDP training step."""
        batch_size = batch.obs.shape[0]
        rng, t_rng, eps_rng = jax.random.split(rng, 3)
        
        # 1. Update Value function V via expectile regression
        target_q = state["critic"].apply_fn(
            state["target_critic"],
            batch.obs,
            batch.act,
        )
        min_target_q = jnp.min(target_q, axis=-1)
        
        def value_loss_fn(value_params):
            v_pred = state["value"].apply_fn(value_params, batch.obs)
            diff = min_target_q - v_pred
            weight = jnp.where(diff > 0, self.algo_cfg.expectile, 1.0 - self.algo_cfg.expectile)
            value_loss = jnp.mean(weight * (diff ** 2))
            info = {
                "v_mean": jnp.mean(v_pred),
                "v_std": jnp.std(v_pred),
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
        
        # 3. Update Diffusion Policy via Advantage-Weighted Score Matching
        current_v = state["value"].apply_fn(value_state.params, batch.obs)
        advantage = min_target_q - current_v
        adv_weights = jnp.exp(advantage / self.algo_cfg.beta)
        adv_weights = jnp.minimum(adv_weights, 100.0)
        adv_weights = jax.lax.stop_gradient(adv_weights)
        
        t = jax.random.randint(t_rng, (batch_size,), 0, self.n_timesteps)
        eps = jax.random.normal(eps_rng, batch.act.shape)
        alpha_bar_t = self.schedule.alphas_cumprod[t][:, None]
        noisy_act = jnp.sqrt(alpha_bar_t) * batch.act + jnp.sqrt(1.0 - alpha_bar_t) * eps
        
        def actor_loss_fn(actor_params):
            pred_eps = state["actor"].apply_fn(actor_params, batch.obs, noisy_act, t)
            sq_err = jnp.sum((pred_eps - eps) ** 2, axis=-1)
            diff_loss = jnp.mean(adv_weights * sq_err)
            info = {
                "diff_loss_unweighted": jnp.mean(sq_err),
                "actor_loss": diff_loss,
            }
            return diff_loss, info

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
            "diff_loss_unweighted": actor_info["diff_loss_unweighted"],
            "v_mean": value_info["v_mean"],
            "v_std": value_info["v_std"],
            "q_mean": critic_info["q_mean"],
            "q_std": critic_info["q_std"],
            "td_target_mean": critic_info["td_target_mean"],
            "adv_weights_mean": jnp.mean(adv_weights),
            "adv_weights_max": jnp.max(adv_weights),
        }
        
        return new_state, metrics

    @partial(jax.jit, static_argnums=(0,))
    def eval_step(self, state: Dict[str, Any], obs: jnp.ndarray) -> jnp.ndarray:
        """Deterministic evaluation step.
        
        Args:
            state: Algorithm state
            obs: Observation [obs_dim] or [batch_size, obs_dim]
            
        Returns:
            Action [act_dim] or [batch_size, act_dim]
        """
        is_single = obs.ndim == 1
        if is_single:
            obs = obs[None, :]
            
        actions = self._sample_actions(
            state["actor"].params,
            obs,
            rng=state.get("rng", jax.random.PRNGKey(0)),
            deterministic=True,
        )
        
        if is_single:
            return actions[0]
        return actions
