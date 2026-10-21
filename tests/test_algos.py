"""Unit tests for offline and online RL algorithms."""

import pytest
import jax
import jax.numpy as jnp

from orlax.core.cfg import TrainingCfg, ModelCfg, AlgoCfg
from orlax.core.types import Batch
from orlax.algos import ALGORITHMS


@pytest.fixture
def dummy_batch():
    batch_size = 32
    obs_dim = 8
    act_dim = 3
    
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    
    return Batch(
        obs=jax.random.normal(k1, (batch_size, obs_dim)),
        act=jax.random.uniform(k2, (batch_size, act_dim), minval=-1.0, maxval=1.0),
        reward=jax.random.normal(k3, (batch_size,)),
        next_obs=jax.random.normal(k4, (batch_size, obs_dim)),
        done=jnp.zeros((batch_size,), dtype=jnp.float32),
    )


@pytest.mark.parametrize(
    "algo_name",
    [
        "bc",
        "cql",
        "iql",
        "sac",
        "ppo",
        "td3_bc",
        "bcq",
        "crr",
        "plas",
        "edac",
        "mopo",
        "rambo",
        "combo",
        "mobile",
        "mcq",
        "bremen",
        "diffusion_ql",
        "edp",
        "xql",
    ],
)
def test_algo_lifecycle(algo_name, dummy_batch):
    obs_dim = 8
    act_dim = 3
    
    cfg = TrainingCfg(algo=algo_name, env_name="dummy", batch_size=32, lr=1e-3)
    model_cfg = ModelCfg(hidden_dims=(64, 64), activation="relu")
    algo_cfg = AlgoCfg()
    
    algo_cls = ALGORITHMS[algo_name]
    algo = algo_cls(obs_dim=obs_dim, act_dim=act_dim, cfg=cfg, model_cfg=model_cfg, algo_cfg=algo_cfg)
    
    # 1. Initialize
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    state = algo.init(init_rng)
    
    assert "actor" in state
    assert state["step"] == 0
    
    # 2. Update step
    rng, update_rng = jax.random.split(rng)
    new_state, metrics = algo.update(state, dummy_batch, update_rng)
    
    assert new_state["step"] == 1
    assert isinstance(metrics, dict)
    for k, v in metrics.items():
        assert jnp.isfinite(v), f"Metric {k} is not finite: {v}"
        
    # 3. Second update step
    rng, update_rng2 = jax.random.split(rng)
    new_state2, metrics2 = algo.update(new_state, dummy_batch, update_rng2)
    assert new_state2["step"] == 2
    
    # 4. Evaluation step
    single_obs = jnp.zeros((obs_dim,))
    act_single = algo.eval_step(new_state2, single_obs)
    assert act_single.shape == (act_dim,)
    
    batch_obs = jnp.zeros((5, obs_dim))
    act_batch = algo.eval_step(new_state2, batch_obs)
    assert act_batch.shape == (5, act_dim)
