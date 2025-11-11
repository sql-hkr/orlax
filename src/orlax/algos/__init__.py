"""Offline RL algorithms."""

from .bc import BC
from .cql import CQL
from .iql import IQL

__all__ = [
    "BC",
    "CQL",
    "IQL",
]

# Algorithm registry for CLI
ALGORITHMS = {
    "bc": BC,
    "cql": CQL,
    "iql": IQL,
}
