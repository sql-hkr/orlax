"""Ensemble-Diversified Actor Critic (EDAC) algorithm implementation."""

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


class EDAC:
    r"""Ensemble-Diversified Actor Critic (EDAC) for offline RL.
    
    EDAC uses an ensemble of Q-critics and regularizes the diversity of their gradients
    with respect to policy actions, preventing common overestimation errors on OOD actions:
    
    $$\min_{\phi} L_{TD}(\phi) + \eta \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi} \left[ \sum_{i \neq j} \langle \nabla_a Q_{\phi_i}(s, a), \nabla_a Q_{\phi_j}(s, a) \rangle \right]$$
    
    Reference: An et al., "Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble" (NeurIPS 2021)
    https://arxiv.org/abs/2110.01548
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
        
        # Default ensemble size: 4 or higher (e.g. 10 in paper)
        self.n_critics = max(algo_cfg.n_critics, 4)
        self.eta = algo_cfg.eta  # Diversity regularization weight
        
        self.actor = Actor(
            action_dim=act_dim,
            hidden_dims=model_cfg.hidden_dims,
            activation=model_cfg.activation,
        )
        
        self.critic = Critic(
            hidden_dims=model_cfg.hidden_dims,
            activation=model_cfg.activation,
            num_critics=self.n_critics,
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
    
    def _sample_actions(
        self,
        actor_params: Any,
        obs: jnp.ndarray,
        rng: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
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
        """Perform one EDAC training step."""
        rng, critic_rng, actor_rng, div_rng = jax.random.split(rng, 4)
        alpha = jnp.exp(state["log_alpha"])
        
        # 1. Critic Update with Gradient Diversity Penalty
        def critic_loss_fn(critic_params):
            next_actions, next_log_probs = self._sample_actions(state["actor"].params, batch.next_obs, critic_rng)
            target_q = state["critic"].apply_fn(state["target_critic"], batch.next_obs, next_actions)
            min_target_q = jnp.min(target_q, axis=-1) - alpha * next_log_probs
            
            td_target = batch.reward + self.algo_cfg.gamma * (1.0 - batch.done) * min_target_q
            td_target = jax.lax.stop_gradient(td_target)
            
            q_values = state["critic"].apply_fn(critic_params, batch.obs, batch.act)
            td_loss = jnp.mean((q_values - td_target[:, None]) ** 2)
            
            # Diversity penalty on policy actions
            policy_actions, _ = self._sample_actions(state["actor"].params, batch.obs, div_rng)
            
            # Compute action gradients for each critic: [batch_size, n_critics, act_dim]
            def single_q_grad(o, a, i):
                def q_val(act_vec):
                    return self.critic.apply(critic_params, o[None, :], act_vec[None, :])[0, i]
                return jax.grad(q_val)(a)
            
            grads = jax.vmap(
                lambda o, a: jax.vmap(lambda i: single_q_grad(o, a, i))(jnp.arange(self.n_critics))
            )(batch.obs, policy_actions)
            
            # Normalize gradients
            norms = jnp.linalg.norm(grads, axis=-1, keepdims=True) + 1e-8
            normalized_grads = grads / norms
            
            # Pairwise inner products: [batch_size, n_critics, n_critics]
            inner_products = jnp.matmul(normalized_grads, jnp.swapaxes(normalized_grads, -1, -2))
            # Subtract diagonal (self-inner products equal 1)
            diversity_loss = (jnp.sum(inner_products, axis=(-1, -2)) - self.n_critics) / (self.n_critics * (self.n_critics - 1))
            diversity_loss = jnp.mean(diversity_loss)
            
            total_critic_loss = td_loss + self.eta * diversity_loss
            
            info = {
                "td_loss": td_loss,
                "diversity_loss": diversity_loss,
                "q_mean": jnp.mean(q_values),
            }
            return total_critic_loss, info
        
        (critic_loss, critic_info), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(state["critic"].params)
        critic_state = state["critic"].apply_gradients(grads=critic_grads)
        
        # 2. Actor Update
        def actor_loss_fn(actor_params):
            actions, log_probs = self._sample_actions(actor_params, batch.obs, actor_rng)
            q_values = state["critic"].apply_fn(critic_state.params, batch.obs, actions)
            min_q = jnp.min(q_values, axis=-1)
            actor_loss = jnp.mean(alpha * log_probs - min_q)
            
            info = {"actor_q_mean": jnp.mean(min_q), "entropy": -jnp.mean(log_probs)}
            return actor_loss, info
        
        (actor_loss, actor_info), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(state["actor"].params)
        actor_state = state["actor"].apply_gradients(grads=actor_grads)
        
        # 3. Alpha Temperature Update
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
            "diversity_loss": critic_info["diversity_loss"],
            "actor_loss": actor_loss,
            "q_mean": critic_info["q_mean"],
            "alpha": jnp.exp(new_log_alpha),
        }
        return new_state, metrics
    
    def eval_step(self, state: Dict[str, Any], obs: jnp.ndarray) -> jnp.ndarray:
        mean, _ = state["actor"].apply_fn(state["actor"].params, obs)
        return mean
