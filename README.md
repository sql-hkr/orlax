# ORLax - Offline Reinforcement Learning with JAX

**ORLax** is an extensible, research-friendly offline reinforcement learning framework built with JAX, Flax, and Optax. It provides clean, typed APIs optimized for editor autocompletion, modular algorithm implementations, and production-ready features like WandB logging and GPU acceleration.

## Features

- 🔥 **Modern JAX Stack**: Built on JAX, Flax, and Optax for high-performance GPU/TPU training
- 📦 **Modular Design**: Clean separation of concerns with pluggable algorithms, models, and datasets
- 🎯 **Type-Safe**: Comprehensive type hints with dataclasses instead of dict-heavy patterns
- 📊 **Built-in Logging**: WandB integration with terminal progress bars (tqdm)
- 🚀 **Production-Ready**: Checkpointing, multi-device training, and reproducible experiments
- 🧪 **Research-Friendly**: Clear interfaces, and easy extensibility

## Algorithms

- **BC** (Behavioral Cloning) - Supervised learning from expert demonstrations
- **CQL** (Conservative Q-Learning) - Conservative offline RL with Q-value penalties
- **IQL** (Implicit Q-Learning) - Expectile regression-based offline RL

## Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/orlax.git
cd orlax

# Install with uv
uv sync
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/orlax.git
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
