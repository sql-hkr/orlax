"""Offline RL algorithms."""

from .bc import BC
from .cql import CQL
from .iql import IQL
from .sac import SAC
from .ppo import PPO
from .td3_bc import TD3BC, TD3_BC
from .bcq import BCQ
from .crr import CRR
from .plas import PLAS
from .edac import EDAC
from .mopo import MOPO
from .rambo import RAMBO
from .combo import COMBO
from .mobile import MOBILE
from .mcq import MCQ
from .bremen import BREMEN
from .diffusion_ql import DiffusionQL
from .edp import EDP
from .xql import XQL

__all__ = [
    "BC",
    "CQL",
    "IQL",
    "SAC",
    "PPO",
    "TD3BC",
    "TD3_BC",
    "BCQ",
    "CRR",
    "PLAS",
    "EDAC",
    "MOPO",
    "RAMBO",
    "COMBO",
    "MOBILE",
    "MCQ",
    "BREMEN",
    "DiffusionQL",
    "EDP",
    "XQL",
]

# Algorithm registry for CLI
ALGORITHMS = {
    "bc": BC,
    "cql": CQL,
    "iql": IQL,
    "sac": SAC,
    "ppo": PPO,
    "td3_bc": TD3BC,
    "td3bc": TD3BC,
    "td3+bc": TD3BC,
    "bcq": BCQ,
    "crr": CRR,
    "plas": PLAS,
    "edac": EDAC,
    "mopo": MOPO,
    "rambo": RAMBO,
    "combo": COMBO,
    "mobile": MOBILE,
    "mcq": MCQ,
    "bremen": BREMEN,
    "diffusion_ql": DiffusionQL,
    "diffusion-ql": DiffusionQL,
    "diffusionql": DiffusionQL,
    "dql": DiffusionQL,
    "edp": EDP,
    "efficient_diffusion_policy": EDP,
    "xql": XQL,
    "extreme_q_learning": XQL,
}
