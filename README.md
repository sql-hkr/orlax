# ORLax - Offline Reinforcement Learning with JAX

[![PyPI version](https://img.shields.io/pypi/v/orlax)](https://pypi.org/project/orlax/)
[![License](https://img.shields.io/github/license/sql-hkr/orlax)](LICENSE)

**ORLax** is an extensible, research-friendly offline reinforcement learning framework built with JAX, Flax, and Optax. It provides clean, typed APIs optimized for editor autocompletion, modular algorithm implementations, and production-ready features like WandB logging and GPU acceleration.

## Features

- 🔥 **Modern JAX Stack**: Built on JAX, Flax, and Optax for high-performance GPU/TPU training
- 📦 **Modular Design**: Clean separation of concerns with pluggable algorithms, models, and datasets
- 🎯 **Type-Safe**: Comprehensive type hints with dataclasses instead of dict-heavy patterns
- 📊 **Built-in Logging**: WandB integration with terminal progress bars (tqdm)
- 🚀 **Production-Ready**: Checkpointing, multi-device training, and reproducible experiments
- 🧪 **Research-Friendly**: Clear interfaces, and easy extensibility

## Algorithms

Implemented in modular JAX/Flax, ordered chronologically by original paper publication / arXiv submission:

- **BC** (Behavioral Cloning, 1989) - Supervised policy learning from demonstration data
- **PPO** (Proximal Policy Optimization, Jul 2017) - Clipped surrogate objective policy gradient
- **SAC** (Soft Actor-Critic, Jan 2018) - Off-policy maximum entropy actor-critic
- **BCQ** (Batch-Constrained deep Q-learning, Dec 2018) - Generative action manifold constraint via CVAE
- **MOPO** (Model-based Offline Policy Optimization, May 2020) - Dynamics ensemble with uncertainty reward penalty
- **CQL** (Conservative Q-Learning, Jun 2020) - Conservative offline RL with Q-value penalties
- **CRR** (Critic Regularized Regression, Jun 2020) - Advantage-weighted policy regression
- **BREMEN** (Behavior-Regularized Model-Ensemble, Jul 2020) - Imaginary rollouts with empirical behavior regularization
- **PLAS** (Policy in Latent Action Space, Nov 2020) - Policy optimization inside CVAE latent space
- **COMBO** (Conservative Offline Model-Based Policy Optimization, Feb 2021) - Model rollouts with conservative Q-regularization
- **TD3+BC** (Twin Delayed DDPG + Behavioral Cloning, Jun 2021) - Minimalist offline RL with BC regularization
- **EDAC** (Ensemble-Diversified Actor Critic, Oct 2021) - Q-ensemble with gradient diversity regularization
- **IQL** (Implicit Q-Learning, Oct 2021) - Expectile regression-based offline RL avoiding OOD queries
- **RAMBO** (Robust Adversarial Model-Based Offline RL, Apr 2022) - Pessimistic adversarial dynamics modeling
- **MCQ** (Masked Conservative Q-Learning, Jun 2022) - Masked out-of-distribution conservative Q-learning
- **Diffusion-QL** (Diffusion Q-Learning, Aug 2022) - Continuous diffusion policy with Q-learning guidance
- **XQL** (Extreme Q-Learning, Mar 2023) - Gumbel regression on value function for optimal offline RL
- **MOBILE** (Model-Based Offline Policy Optimization with Local Uncertainty Regularization, Mar 2023) - Local Bellman uncertainty penalty
- **EDP** (Efficient Diffusion Policy, May 2023) - Decoupled diffusion policy with advantage-weighted score matching

## Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/sql-hkr/orlax.git
cd orlax

# Install with uv
uv sync
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/sql-hkr/orlax.git
cd orlax

# Install in editable mode
pip install -e .
```

### GPU Support

For CUDA support, install JAX with CUDA:

```bash
# For CUDA 12
pip install --upgrade "jax[cuda12]"
```

## Quick Start

### Training

```bash
# Train IQL on Hopper-Medium
uv run orlax-train --config configs/iql_hopper.toml
```

### Verification & Testing

```bash
# Run multi-step training verification across all 19 algorithms (Apple Metal GPU / CUDA / CPU)
uv run python scripts/verify_all_training.py

# Run test suite
uv run pytest
```

## Citation

If you use ORLax in your research, please cite:

```bibtex
@software{orlax2025,
  title = {ORLax: Offline Reinforcement Learning with JAX},
  author = {sql-hkr},
  year = {2025},
  url = {https://github.com/sql-hkr/orlax}
}
```

## Acknowledgments

- Built with [JAX](https://github.com/google/jax), [Flax](https://github.com/google/flax), and [Optax](https://github.com/deepmind/optax)
- Offline RL datasets from [Minari](https://github.com/Farama-Foundation/Minari) (successor to D4RL)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feat/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feat/amazing-feature`)
5. Open a Pull Request

## Contact

- **Author**: sql-hkr
- **Email**: sql.hkr@gmail.com
- **GitHub**: [@sql-hkr](https://github.com/sql-hkr)
- **Issues**: [GitHub Issues](https://github.com/sql-hkr/orlax/issues)

---

**Note**: This software is under active development. API stability is not guaranteed until version 1.0.0.
