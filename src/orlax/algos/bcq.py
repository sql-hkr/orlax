"""Batch-Constrained deep Q-learning (BCQ) algorithm implementation."""

from typing import Any, Dict
from functools import partial
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from ..core.types import Batch
from ..core.cfg import TrainingCfg, ModelCfg, AlgoCfg
from ..models.nets import Actor, Critic, ConditionalVAE


class BCQ:
    r"""Batch-Constrained deep Q-learning for continuous control.
    
    BCQ restricts the action space to state-action pairs similar to the dataset
    by combining a Conditional VAE generative model with an action perturbation network:
    
    $$a = \text{decode}(s, z) + \xi_\phi(s, \text{decode}(s, z))$$
    
    Reference: Fujimoto et al., "Off-Policy Deep Reinforcement Learning without Exploration" (ICML 2019)
    https://arxiv.org/abs/1812.02900
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
        
        self.phi = 0.05  # Perturbation scale max
        
        self.vae = ConditionalVAE(
            action_dim=act_dim,
            latent_dim=algo_cfg.latent_dim,
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
        rng, vae_rng, actor_rng, critic_rng = jax.random.split(rng, 4)
        
        dummy_obs = jnp.zeros((1, self.obs_dim))
        dummy_act = jnp.zeros((1, self.act_dim))
        
        vae_params = self.vae.init(vae_rng, dummy_obs, dummy_act)
        
        actor_params = self.actor.init(actor_rng, dummy_obs)
        critic_params = self.critic.init(critic_rng, dummy_obs, dummy_act)
        
        vae_state = TrainState.create(
            apply_fn=self.vae.apply,
            params=vae_params,
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
        
        target_actor_params = jax.tree.map(lambda x: x, actor_params)
        target_critic_params = jax.tree.map(lambda x: x, critic_params)
        
        return {
            "vae": vae_state,
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
        """Perform one BCQ training step."""
        rng, vae_rng, sample_rng = jax.random.split(rng, 3)
        
        # 1. VAE Update (ELBO = Reconstruction + KL)
        def vae_loss_fn(vae_params):
            mean, log_std = self.vae.apply(vae_params, batch.obs, batch.act)
            std = jnp.exp(log_std)
            eps = jax.random.normal(vae_rng, mean.shape)
            z = mean + std * eps
            
            recon_act = self.vae.apply(vae_params, batch.obs, z, method=self.vae.decode)
            recon_loss = jnp.mean((recon_act - batch.act) ** 2)
            
            # KL divergence to standard normal
            kl_loss = -0.5 * jnp.mean(1.0 + 2.0 * log_std - mean ** 2 - std ** 2)
            vae_loss = recon_loss + 0.5 * kl_loss
            
            return vae_loss, {"recon_loss": recon_loss, "kl_loss": kl_loss}
        
        (vae_loss, vae_info), vae_grads = jax.value_and_grad(vae_loss_fn, has_aux=True)(state["vae"].params)
        vae_state = state["vae"].apply_gradients(grads=vae_grads)
        
        # 2. Critic Update
        def critic_loss_fn(critic_params):
            # Sample candidate action from VAE
            z_sample = jax.random.normal(sample_rng, (batch.obs.shape[0], self.algo_cfg.latent_dim))
            z_sample = jnp.clip(z_sample, -0.5, 0.5)
            next_cand_act = self.vae.apply(vae_state.params, batch.next_obs, z_sample, method=self.vae.decode)
            
            # Perturb with target actor
            perturbation, _ = self.actor.apply(state["target_actor"], batch.next_obs)
            next_act = jnp.clip(next_cand_act + self.phi * perturbation, -1.0, 1.0)
            
            target_q = state["critic"].apply_fn(state["target_critic"], batch.next_obs, next_act)
            min_q = jnp.min(target_q, axis=-1)
            max_q = jnp.max(target_q, axis=-1)
            # Soft clipped double Q target (0.75 min + 0.25 max)
            soft_target_q = 0.75 * min_q + 0.25 * max_q
            
            td_target = batch.reward + self.algo_cfg.gamma * (1.0 - batch.done) * soft_target_q
            td_target = jax.lax.stop_gradient(td_target)
            
            q_values = state["critic"].apply_fn(critic_params, batch.obs, batch.act)
            critic_loss = jnp.mean((q_values - td_target[:, None]) ** 2)
            
            return critic_loss, {"q_mean": jnp.mean(q_values), "target_q_mean": jnp.mean(td_target)}
        
        (critic_loss, critic_info), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(state["critic"].params)
        critic_state = state["critic"].apply_gradients(grads=critic_grads)
        
        # 3. Perturbation Actor Update
        def actor_loss_fn(actor_params):
            z_zero = jnp.zeros((batch.obs.shape[0], self.algo_cfg.latent_dim))
            base_act = self.vae.apply(vae_state.params, batch.obs, z_zero, method=self.vae.decode)
            perturbation, _ = self.actor.apply(actor_params, batch.obs)
            act = jnp.clip(base_act + self.phi * perturbation, -1.0, 1.0)
            
            q_vals = state["critic"].apply_fn(critic_state.params, batch.obs, act)
            q1 = q_vals[:, 0]
            actor_loss = -jnp.mean(q1)
            return actor_loss, {"actor_q": jnp.mean(q1)}
        
        (actor_loss, actor_info), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(state["actor"].params)
        actor_state = state["actor"].apply_gradients(grads=actor_grads)
        
        # 4. Target Polyak Averaging
        target_critic = jax.tree.map(
            lambda t, o: self.algo_cfg.tau * o + (1.0 - self.algo_cfg.tau) * t,
            state["target_critic"],
            critic_state.params,
        )
        target_actor = jax.tree.map(
            lambda t, o: self.algo_cfg.tau * o + (1.0 - self.algo_cfg.tau) * t,
            state["target_actor"],
            actor_state.params,
        )
        
        new_state = {
            "vae": vae_state,
            "actor": actor_state,
            "critic": critic_state,
            "target_actor": target_actor,
            "target_critic": target_critic,
            "step": state["step"] + 1,
            "rng": rng,
        }
        
        metrics = {
            "vae_loss": vae_loss,
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "recon_loss": vae_info["recon_loss"],
            "kl_loss": vae_info["kl_loss"],
            "q_mean": critic_info["q_mean"],
        }
        return new_state, metrics
    
    @partial(jax.jit, static_argnums=(0,))
    def eval_step(self, state: Dict[str, Any], obs: jnp.ndarray) -> jnp.ndarray:
        """Deterministic action for evaluation."""
        obs_batch = obs if obs.ndim == 2 else obs[None, :]
        z = jnp.zeros((obs_batch.shape[0], self.algo_cfg.latent_dim))
        base_act = self.vae.apply(state["vae"].params, obs_batch, z, method=self.vae.decode)
        perturbation, _ = self.actor.apply(state["actor"].params, obs_batch)
        act = jnp.clip(base_act + self.phi * perturbation, -1.0, 1.0)
        return act if obs.ndim == 2 else act[0]
