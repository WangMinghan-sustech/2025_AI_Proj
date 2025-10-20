"""
PyTorch DQN for CartPole (Gymnasium)
------------------------------------
- Uses an online Q-network and a target Q-network
- Vectorized replay updates for speed
- ε-greedy exploration
- Compatible with train.py in this project structure
"""

from __future__ import annotations
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# Default Hyperparameters
# -----------------------------
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 32
MEMORY_SIZE = 50_000
INITIAL_EXPLORATION_STEPS = 1_000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
TARGET_UPDATE_STEPS = 500


class QNet(nn.Module):
    """Simple MLP for Q(s, a)."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )
        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """FIFO replay buffer storing numpy arrays; tensors are created on sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=capacity)

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool):
        # mask = 0 if done else 1
        self.buf.append((s, a, r, s2, 0.0 if done else 1.0))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, m = zip(*batch)
        return (
            np.stack(s, axis=0),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(s2, axis=0),
            np.array(m, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


@dataclass
class DQNConfig:
    gamma: float = GAMMA
    lr: float = LR
    batch_size: int = BATCH_SIZE
    memory_size: int = MEMORY_SIZE
    initial_exploration: int = INITIAL_EXPLORATION_STEPS
    eps_start: float = EPS_START
    eps_end: float = EPS_END
    eps_decay: float = EPS_DECAY
    target_update: int = TARGET_UPDATE_STEPS
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DQNSolver:
    """
    PyTorch DQN agent.
    Exposes: act(), remember(), experience_replay(), save(), load(), update_target()
    """

    def __init__(self, observation_space: int, action_space: int, cfg: DQNConfig | None = None):
        self.obs_dim = observation_space
        self.act_dim = action_space
        self.cfg = cfg or DQNConfig()

        self.device = torch.device(self.cfg.device)

        self.online = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.target = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.update_target(hard=True)

        self.optim = optim.Adam(self.online.parameters(), lr=self.cfg.lr)
        self.memory = ReplayBuffer(self.cfg.memory_size)

        self.steps = 0
        self.exploration_rate = self.cfg.eps_start

    # -----------------------------
    # API
    # -----------------------------
    def act(self, state_np: np.ndarray) -> int:
        """
        ε-greedy action. `state_np` is shape (1, obs_dim) numpy array.
        """
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.act_dim)

        with torch.no_grad():
            s = torch.as_tensor(state_np, dtype=torch.float32, device=self.device)
            q = self.online(s)  # [1, act_dim]
            a = int(torch.argmax(q, dim=1).item())
        return a

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.memory.push(state, action, reward, next_state, done)

    def experience_replay(self):
        """
        Vectorized replay update with target network.
        """
        if len(self.memory) < max(self.cfg.batch_size, self.cfg.initial_exploration):
            # Exploration warm-up or not enough data
            self._decay_eps()  # still decay slowly each step
            return

        s, a, r, s2, m = self.memory.sample(self.cfg.batch_size)

        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a_t = torch.as_tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)  # [B,1]
        r_t = torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)  # [B,1]
        s2_t = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        m_t = torch.as_tensor(m, dtype=torch.float32, device=self.device).unsqueeze(1)  # [B,1]

        # Q(s,a) from online net
        q_sa = self.online(s_t).gather(1, a_t)  # [B,1]

        # Target: r + mask * gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            q_next = self.target(s2_t).max(dim=1, keepdim=True)[0]  # [B,1]
            target = r_t + m_t * self.cfg.gamma * q_next

        loss = nn.functional.mse_loss(q_sa, target)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # Soft ε decay per training step
        self._decay_eps()

        # Periodic hard target update
        if self.steps % self.cfg.target_update == 0:
            self.update_target(hard=True)

    def update_target(self, hard: bool = True, tau: float = 0.005):
        if hard:
            self.target.load_state_dict(self.online.state_dict())
        else:
            # Polyak (soft) update
            with torch.no_grad():
                for p_t, p in zip(self.target.parameters(), self.online.parameters()):
                    p_t.data.mul_(1 - tau).add_(tau * p.data)

    def save(self, path: str):
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "cfg": self.cfg.__dict__,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        # cfg could be reloaded if needed

    # -----------------------------
    # Helpers
    # -----------------------------
    def _decay_eps(self):
        self.exploration_rate = max(self.cfg.eps_end, self.exploration_rate * self.cfg.eps_decay)
        self.steps += 1