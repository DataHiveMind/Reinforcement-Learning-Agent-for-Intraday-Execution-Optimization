# env/orderbook_env.py
import gym
import numpy as np
from gym import spaces
from typing import Optional, Tuple, Dict, Any

class OrderBookEnv(gym.Env):
    """
    Gym env simulating a simple LOB with latency.
    State: [mid, spread, imbalance, inventory, time_left]
    Actions: wait, MktBuy, MktSell, LimBuy, LimSell
    Reward: -slippage - latency_penalty
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        tick_data: np.ndarray,
        max_steps: int = 100,
        latency_mean: float = 1.0,
        latency_std: float = 0.5,
        latency_cost: float = 0.1
    ):
        super().__init__()
        self.tick_data = tick_data
        self.max_steps = max_steps
        self.latency_mean = latency_mean
        self.latency_std = latency_std
        self.latency_cost = latency_cost

        self.current_step = 0
        self.inventory = 0.0

        # Observation: mid_price, spread, imbalance, inventory, time_left
        low  = np.array([0, 0, -1, -np.inf, 0], dtype=np.float32)
        high = np.array([np.inf, np.inf, 1, np.inf, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Discrete actions: wait, market buy, market sell, limit buy, limit sell
        self.action_space = spaces.Discrete(5)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Handle seeding
        super().reset(seed=seed)

        # Reset internal state
        self.current_step = 0
        self.inventory = 0.0

        # Return initial observation and empty info dict
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> np.ndarray:
        mid, spread, imb = self.tick_data[self.current_step]
        time_left = 1.0 - (self.current_step / self.max_steps)
        return np.array(
            [mid, spread, imb, self.inventory, time_left],
            dtype=np.float32
        )

    def step(self, action: int) -> Tuple[
        Optional[np.ndarray],
        float,
        bool,
        bool,
        Dict[str, Any]
    ]:
        """
        Returns:
          obs, reward, terminated, truncated, info
        """
        mid, spread, imb = self.tick_data[self.current_step]

        # Execution logic
        if action == 1:       # market buy
            qty, cost = 1.0, mid + spread / 2
        elif action == 2:     # market sell
            qty, cost = -1.0, mid - spread / 2
        elif action == 3:     # limit buy
            qty, cost = 0.5, mid - spread / 4
        elif action == 4:     # limit sell
            qty, cost = -0.5, mid + spread / 4
        else:                 # wait
            qty, cost = 0.0, mid

        # Latency simulation
        latency = max(0.0, np.random.normal(self.latency_mean, self.latency_std))
        lat_penalty = latency * self.latency_cost

        self.inventory += qty
        slippage = abs(cost - mid)
        reward = -slippage - lat_penalty

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False  # you could set a separate time limit

        obs = self._get_obs() if not terminated else None
        info = {
            'exec_cost': cost,
            'inventory': self.inventory,
            'latency': latency,
            'slippage': slippage
        }

        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass
