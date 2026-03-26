from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore


@dataclass
class Args:
    exp_name: str = "ppo_cart_pole"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = True

    # Algorithm specific arguments
    total_timesteps: int = 500000
    learning_rate: float = 2.5e-4
    num_envs: int = 4
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None

    # Computed at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


cs = ConfigStore.instance()
cs.store(name="config", node=Args)