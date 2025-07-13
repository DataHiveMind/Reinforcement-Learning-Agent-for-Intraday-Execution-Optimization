# Reinforcement-Learning-Agent-for-Intraday-Execution-Optimization

An advanced RL framework for intraday execution optimization with  
latency simulation, double‐DQN, and quantitative risk analytics.

## 🚀 Features

- Custom Gym env with latency sampling & slippage cost  
- Double DQN with target sync, epsilon decay  
- TensorBoard logging of reward, latency, and more  
- Monte Carlo backtests: PnL distribution, VaR & CVaR  
- CLI modes: `train`, `eval`, `eval_mc`

## 📂 Structure
rl_execution_optimizer/ 
├── env/ # Latency‐aware LOB simulator 
├── agent/ # DoubleDQN (and PG) implementations 
├── utils/ # Metrics, logger, visualizer 
├── train/ # Training loop with TB logging 
├── eval/ # Single‐run & MonteCarlo backtests 
├── main.py # Entry point 
├── requirements.txt 
└── README.md


## 📦 Install

```bash
pip install -r requirements.txt

## 🛠 Usage
Train agent:

```bash
python main.py --mode train --data data/mock_tick_data.csv

## Evaluate a single episode:
```bash
python main.py --mode eval  --data data/mock_tick_data.csv --model double_dqn.pth

## 📊 Output
TensorBoard logs under runs/exp

PnL distribution plots via utils/visualizer.py

VaR & CVaR risk metrics