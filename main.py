import argparse
import pandas as pd
import torch
from train.train_agent import train_dqn
from eval.backtest import single_run, monte_carlo_backtest

def load_tick_data(path):
    df = pd.read_csv(path)
    return df[['mid_price','spread','imbalance']].to_numpy()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['train','eval','eval_mc'], required=True)
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--model', type=str)
    p.add_argument('--sims',  type=int, default=200)
    p.add_argument('--sigma', type=float, default=0.01)
    args = p.parse_args()

    data = load_tick_data(args.data)

    if args.mode == 'train':
        agent = train_dqn(data, episodes=500)
        torch.save(agent.online_net.state_dict(), 'double_dqn.pth')
        print("Training done. Model saved to double_dqn.pth")

    else:
        # load agent
        from agent.dqn_agent import DQNAgent
        agent = DQNAgent(state_dim=5, action_dim=5)
        agent.online_net.load_state_dict(torch.load(args.model))
        if args.mode == 'eval':
            bench, slip, fill = single_run(agent, data)
            print(f"Bench VWAP: {bench:.4f}, Slippage: {slip:.4f}, Fill: {fill:.2%}")
        else:
            stats = monte_carlo_backtest(agent, data,
                                        sims=args.sims, sigma=args.sigma)
            print("Monte Carlo Results:")
            print(f"Mean PnL: {stats['pnl_mean']:.4f}")
            print(f"Std  PnL: {stats['pnl_std']:.4f}")
            print(f"VaR(5%): {stats['VaR_5%']:.4f}, CVaR(5%): {stats['CVaR_5%']:.4f}")
            print("Distribution of PnL:", stats['distribution'])