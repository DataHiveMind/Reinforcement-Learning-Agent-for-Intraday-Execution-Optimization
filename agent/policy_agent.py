# agent/policy_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)

class ActorCriticAgent:
    """
    Advantage Actor-Critic agent for continuous training:
      - Uses separate policy (actor) and value (critic) networks
      - Optimizes actor via policy gradient with advantage
      - Optimizes critic via mean-squared error to bootstrapped returns
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        entropy_coef=0.01,
        value_coef=0.5
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.value_net  = ValueNetwork(state_dim).to(self.device)

        self.actor_opt  = optim.Adam(self.policy_net.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.value_net.parameters(), lr=critic_lr)

        self.gamma         = gamma
        self.entropy_coef  = entropy_coef
        self.value_coef    = value_coef

        # storage
        self.log_probs = []
        self.values    = []
        self.rewards   = []
        self.entropies = []

    def select_action(self, state):
        """
        Given a state, sample an action and store log_prob, value, and entropy.
        """
        state_v = torch.FloatTensor(state).to(self.device)
        probs   = self.policy_net(state_v)
        dist    = Categorical(probs)

        action      = dist.sample()
        log_prob    = dist.log_prob(action)
        entropy     = dist.entropy()
        value       = self.value_net(state_v)

        self.log_probs.append(log_prob)
        self.values.append(value)
        self.entropies.append(entropy)

        return action.item()

    def store_reward(self, reward):
        self.rewards.append(torch.tensor(reward, device=self.device))

    def finish_episode(self):
        """
        Compute returns, advantages, and losses, then backprop.
        Clears buffers after updating.
        """
        # compute discounted returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.stack(returns)

        values = torch.stack(self.values)
        log_probs = torch.stack(self.log_probs)
        entropies = torch.stack(self.entropies)

        # advantage = R - V(s)
        advantages = returns - values.detach()

        # actor loss (policy gradient with entropy bonus)
        actor_loss = -(log_probs * advantages).mean() - self.entropy_coef * entropies.mean()

        # critic loss (value MSE)
        critic_loss = F.mse_loss(values, returns)

        # combined loss
        loss = actor_loss + self.value_coef * critic_loss

        # backprop
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()
        self.critic_opt.step()

        # clear buffers
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.entropies.clear()

    def save(self, path_prefix='actor_critic'):
        torch.save(self.policy_net.state_dict(), f"{path_prefix}_policy.pth")
        torch.save(self.value_net.state_dict(),  f"{path_prefix}_value.pth")

    def load(self, path_prefix='actor_critic'):
        self.policy_net.load_state_dict(torch.load(f"{path_prefix}_policy.pth"))
        self.value_net.load_state_dict(torch.load(f"{path_prefix}_value.pth"))
