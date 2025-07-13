# train/train_agent.py

import numpy as np
from tqdm import trange
from typing import Optional

from gym.spaces import Discrete
from env.orderbook_env import OrderBookEnv
from agent.dqn_agent import DQNAgent        # ensure this matches your class name
from agent.policy_agent import ActorCriticAgent
from utils.logger import get_logger, log_metrics


def train_dqn(
    tick_data: np.ndarray,
    episodes: int = 500,
    max_steps: Optional[int] = None
) -> DQNAgent:
    # Guarantee max_steps is an int
    steps = max_steps if max_steps is not None else len(tick_data)
    env = OrderBookEnv(tick_data, max_steps=steps)

    # Confirm action_space.n exists
    assert isinstance(env.action_space, Discrete)
    action_dim: int = env.action_space.n

    agent = DQNAgent(state_dim=5, action_dim=action_dim)
    writer = get_logger('runs/dqn')

    for ep in trange(episodes, desc="Training DQN"):
        # reset returns (obs, info)
        state, _ = env.reset()
        total_reward = 0.0
        latencies = []

        while True:
            # select_action returns an int
            action = int(agent.select_action(state))

            # step returns (obs, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store(
                state,
                action,
                reward,
                None if done else next_state,
                done
            )
            agent.update()

            if done:
                latencies.append(info['latency'])
                total_reward += reward
                break

            state = next_state       # state is np.ndarray
            total_reward += reward
            latencies.append(info['latency'])

        avg_lat = float(np.mean(latencies))
        log_metrics(writer, ep, total_reward, avg_lat)

    writer.close()
    return agent


def train_a2c(
    tick_data: np.ndarray,
    episodes: int = 500,
    max_steps: Optional[int] = None
) -> ActorCriticAgent:
    steps = max_steps if max_steps is not None else len(tick_data)
    env = OrderBookEnv(tick_data, max_steps=steps)

    assert isinstance(env.action_space, Discrete)
    action_dim: int = env.action_space.n

    agent = ActorCriticAgent(state_dim=5, action_dim=action_dim)
    writer = get_logger('runs/a2c')

    for ep in trange(episodes, desc="Training A2C"):
        state, _ = env.reset()
        total_reward = 0.0
        latencies = []

        while True:
            action = int(agent.select_action(state))
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_reward(reward)

            if done:
                agent.finish_episode()
                latencies.append(info['latency'])
                total_reward += reward
                break

            state = next_state
            total_reward += reward
            latencies.append(info['latency'])

        avg_lat = float(np.mean(latencies))
        log_metrics(writer, ep, total_reward, avg_lat)

    writer.close()
    return agent

