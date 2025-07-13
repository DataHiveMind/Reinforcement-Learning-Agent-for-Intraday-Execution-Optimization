# train/train_agent.py

import numpy as np
from tqdm import trange
from env.orderbook_env import OrderBookEnv
from agent.dqn_agent import DQNAgent
from agent.policy_agent import ActorCriticAgent
from utils.logger import get_logger, log_metrics

def train_dqn(tick_data, episodes=500, max_steps=None):
    max_steps = max_steps or len(tick_data)
    env = OrderBookEnv(tick_data, max_steps=max_steps)
    agent = DQNAgent(state_dim=5, action_dim=env.action_space.n)
    writer = get_logger('runs/dqn')

    for ep in trange(episodes, desc="Training DQN"):
        state = env.reset()
        total_reward = 0
        latencies = []

        while True:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward
            latencies.append(info['latency'])
            if done:
                break

        avg_lat = np.mean(latencies)
        log_metrics(writer, ep, total_reward, avg_lat)

    writer.close()
    return agent

def train_a2c(tick_data, episodes=500, max_steps=None):
    max_steps = max_steps or len(tick_data)
    env = OrderBookEnv(tick_data, max_steps=max_steps)
    agent = ActorCriticAgent(state_dim=5, action_dim=env.action_space.n)
    writer = get_logger('runs/a2c')

    for ep in trange(episodes, desc="Training A2C"):
        state = env.reset()
        total_reward = 0
        latencies = []

        while True:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_reward(reward)

            state = next_state
            total_reward += reward
            latencies.append(info['latency'])
            if done:
                agent.finish_episode()
                break

        avg_lat = np.mean(latencies)
        log_metrics(writer, ep, total_reward, avg_lat)

    writer.close()
    return agent

