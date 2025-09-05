# agent/dqn_agent.py

import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class DQNNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.1,
        eps_decay: float = 1e-4,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_sync: int = 1000
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim

        # 1) Build and move online network
        net = DQNNetwork(state_dim, action_dim)
        net.to(self.device)            # in‐place, keeps `net` typed as DQNNetwork
        self.online_net = net

        # 2) Build and move target network
        tgt = DQNNetwork(state_dim, action_dim)
        tgt.to(self.device)
        self.target_net = tgt
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer   = optim.Adam(self.online_net.parameters(), lr=lr)
        self.gamma       = gamma
        self.epsilon     = eps_start
        self.eps_end     = eps_end
        self.eps_decay   = eps_decay
        self.replay_buf  = deque(maxlen=buffer_size)
        self.batch_size  = batch_size
        self.target_sync = target_sync
        self.step_count  = 0

    def select_action(self, state: np.ndarray) -> int:
        # Epsilon‐greedy
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_v = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            qvals = self.online_net(state_v)
        return int(qvals.argmax().item())

    def store(self, s, a, r, s2, done):
        self.replay_buf.append((s, a, r, s2, done))

    def update(self):
        if len(self.replay_buf) < self.batch_size:
            return

        batch = random.sample(self.replay_buf, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # to tensors
        s  = torch.FloatTensor(states).to(self.device)
        a  = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        r  = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        d  = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # mask non‐final next states
        non_final_mask = torch.BoolTensor([st is not None for st in next_states])
        s2_v = torch.FloatTensor(
            [st for st in next_states if st is not None]
        ).to(self.device)

        # current Q
        q_val = self.online_net(s).gather(1, a)

        # Double‐DQN target
        next_actions = self.online_net(s2_v).argmax(dim=1, keepdim=True)
        next_q       = self.target_net(s2_v).gather(1, next_actions)
        target_q     = r[non_final_mask] + self.gamma * next_q * (1 - d[non_final_mask])

        full_target = torch.zeros_like(q_val).to(self.device)
        full_target[non_final_mask] = target_q.detach()

        loss = nn.MSELoss()(q_val, full_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon & sync target network
        self.epsilon = max(self.eps_end, self.epsilon - self.eps_decay)
        self.step_count += 1
        if self.step_count % self.target_sync == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
