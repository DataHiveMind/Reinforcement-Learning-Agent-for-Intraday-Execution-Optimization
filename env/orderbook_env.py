import gym
import numpy as np
from gym import spaces

class OrderBookEnv(gym.Env):
    """
    Gym env simulating a simple LOB with latency.
    State: [mid, spread, imbalance, inventory, time_left]
    Actions: wait, MktBuy, MktSell, LimBuy, LimSell
    Reward: -slippage - latency_penalty
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, tick_data, max_steps=100,
                 latency_mean=1.0, latency_std=0.5, latency_cost=0.1):
        super().__init__()
        self.tick_data = tick_data
        self.max_steps = max_steps
        self.latency_mean = latency_mean
        self.latency_std = latency_std
        self.latency_cost = latency_cost

        self.current_step = 0
        self.inventory = 0

        low = np.array([0, 0, -1, -np.inf, 0], dtype=np.float32)
        high = np.array([np.inf, np.inf, 1, np.inf, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Discrete(5)

    def reset(self):
        self.current_step = 0
        self.inventory = 0
        return self._get_obs()

    def _get_obs(self):
        mid, spread, imb = self.tick_data[self.current_step]
        t_left = 1 - self.current_step / self.max_steps
        return np.array([mid, spread, imb, self.inventory, t_left],
                        dtype=np.float32)

    def step(self, action):
        mid, spread, imb = self.tick_data[self.current_step]

        # simulate execution
        if action == 1:       # market buy
            qty, cost = 1, mid + spread/2
        elif action == 2:     # market sell
            qty, cost = -1, mid - spread/2
        elif action == 3:     # limit buy
            qty, cost = 0.5, mid - spread/4
        elif action == 4:     # limit sell
            qty, cost = -0.5, mid + spread/4
        else:                 # wait
            qty, cost = 0, mid

        # apply latency
        latency = max(0.0, np.random.normal(self.latency_mean, self.latency_std))
        lat_pen = latency * self.latency_cost

        self.inventory += qty
        slippage = abs(cost - mid)
        reward = -slippage - lat_pen

        self.current_step += 1
        done = self.current_step >= self.max_steps
        obs = self._get_obs() if not done else None

        info = {
            'exec_cost': cost,
            'inventory': self.inventory,
            'latency': latency,
            'slippage': slippage
        }
        return obs, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass
