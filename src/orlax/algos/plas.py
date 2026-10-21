"""Policy in Latent Action Space (PLAS) algorithm implementation."""

from typing import Any, Dict
from functools import partial
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from ..core.types import Batch
from ..core.cfg import TrainingCfg, ModelCfg, AlgoCfg
from ..models.nets import Actor, Critic, ConditionalVAE


class PLAS:
    r"""Policy in Latent Action Space (PLAS) for offline RL.
    
    PLAS restricts policy search to the manifold of behaviors observed in the dataset
    by training the policy network to produce latent actions $z \in \mathcal{Z}$, which
    are then passed through a pre-trained or co-trained conditional VAE decoder $G(s, z)$:
    
    $$a = G(s, \pi_\theta(s))$$
    
    Reference: Zhou et al., "PLAS: Latent Action Space for Offline Reinforcement Learning" (CoRL 2020)
    https://arxiv.org/abs/2011.07213
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
        
        self.vae = ConditionalVAE(
            action_dim=act_dim,
            latent_dim=algo_cfg.latent_dim,
            hidden_dims=model_cfg.hidden_dims,
            activation=model_cfg.activation,
        )
        
        # Policy outputs in latent space z
        self.actor = Actor(
            action_dim=algo_cfg.latent_dim,
            hidden_dims=model_cfg.hidden_dims,
            activation=model_cfg.activation,
            max_action=2.0,  # Latent bounds
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
        """Perform one PLAS training step."""
        rng, vae_rng = jax.random.split(rng)
        
        # 1. VAE Update
        def vae_loss_fn(vae_params):
            mean, log_std = self.vae.apply(vae_params, batch.obs, batch.act)
            std = jnp.exp(log_std)
            eps = jax.random.normal(vae_rng, mean.shape)
            z = mean + std * eps
            
            recon_act = self.vae.apply(vae_params, batch.obs, z, method=self.vae.decode)
            recon_loss = jnp.mean((recon_act - batch.act) ** 2)
            kl_loss = -0.5 * jnp.mean(1.0 + 2.0 * log_std - mean ** 2 - std ** 2)
            
            return recon_loss + 0.5 * kl_loss, {"recon_loss": recon_loss, "kl_loss": kl_loss}
        
        (vae_loss, vae_info), vae_grads = jax.value_and_grad(vae_loss_fn, has_aux=True)(state["vae"].params)
        vae_state = state["vae"].apply_gradients(grads=vae_grads)
        
        # 2. Critic Update
        def critic_loss_fn(critic_params):
            # Target latent action
            next_z, _ = self.actor.apply(state["target_actor"], batch.next_obs)
            next_act = self.vae.apply(vae_state.params, batch.next_obs, next_z, method=self.vae.decode)
            
            target_q = state["critic"].apply_fn(state["target_critic"], batch.next_obs, next_act)
            target_q = jnp.min(target_q, axis=-1)
            
            td_target = batch.reward + self.algo_cfg.gamma * (1.0 - batch.done) * target_q
            td_target = jax.lax.stop_gradient(td_target)
            
            q_values = state["critic"].apply_fn(critic_params, batch.obs, batch.act)
            critic_loss = jnp.mean((q_values - td_target[:, None]) ** 2)
            
            return critic_loss, {"q_mean": jnp.mean(q_values), "target_q_mean": jnp.mean(td_target)}
        
        (critic_loss, critic_info), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(state["critic"].params)
        critic_state = state["critic"].apply_gradients(grads=critic_grads)
        
        # 3. Latent Policy Update (maximize Q(s, decode(s, pi(s))))
        def actor_loss_fn(actor_params):
            latent_z, _ = state["actor"].apply_fn(actor_params, batch.obs)
            actions = self.vae.apply(vae_state.params, batch.obs, latent_z, method=self.vae.decode)
            
            q_vals = state["critic"].apply_fn(critic_state.params, batch.obs, actions)
            actor_loss = -jnp.mean(q_vals[:, 0])
            return actor_loss, {"actor_q": -actor_loss}
        
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
            "q_mean": critic_info["q_mean"],
        }
        return new_state, metrics
    
    @partial(jax.jit, static_argnums=(0,))
    def eval_step(self, state: Dict[str, Any], obs: jnp.ndarray) -> jnp.ndarray:
        """Deterministic evaluation action."""
        obs_batch = obs if obs.ndim == 2 else obs[None, :]
        z, _ = state["actor"].apply_fn(state["actor"].params, obs_batch)
        act = self.vae.apply(state["vae"].params, obs_batch, z, method=self.vae.decode)
        return act if obs.ndim == 2 else act[0]
