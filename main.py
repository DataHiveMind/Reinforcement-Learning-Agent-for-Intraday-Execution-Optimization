import argparse
import pandas as pd
import torch
import matplotlib.pyplot as plt
from agent.dqn_agent import DQNAgent
from agent.policy_agent import ActorCriticAgent
from train.train_agent import train_dqn, train_a2c
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
    p.add_argument('--target_qty', type=int, default=10)
    args = p.parse_args()

    data = load_tick_data(args.data)

    if args.mode == 'train':
        from agent.dqn_agent import DQNAgent
        agent1 = train_dqn(data, episodes=500)
        torch.save(agent1.online_net.state_dict(), '/workspaces/Reinforcement-Learning-Agent-for-Intraday-Execution-Optimization/models/DQN/double_dqn.pth')
        print("Training done. Model saved to double_dqn.pth")
        
        agent2 = train_a2c(data, episodes=500)
        agent2.save('/workspaces/Reinforcement-Learning-Agent-for-Intraday-Execution-Optimization/models/A2C/a2c_actor_critic')
        print("Training done. Model saved to a2c_actor_critic_policy.pth and a2c_actor_critic_value.pth")
        agent = agent1
    else:
        # load agent
        from agent.dqn_agent import DQNAgent
        agent = DQNAgent(state_dim=5, action_dim=5)
        agent.online_net.load_state_dict(torch.load(args.model))
        agent.epsilon = 0.0
        
    if args.mode == 'eval':
        bench, slip, fill = single_run(agent, data, target_qty=args.target_qty)
        print(f"Bench VWAP: {bench:.4f}, Slippage: {slip:.4f}, Fill: {fill:.2%}")
    elif args.mode == 'eval_mc':
        stats = monte_carlo_backtest(agent, data,
                                    sims=args.sims, sigma=args.sigma, target_qty=args.target_qty)
        print("Monte Carlo Results:")
        print(f"Mean PnL: {stats['pnl_mean']:.4f}")
        print(f"Std  PnL: {stats['pnl_std']:.4f}")
        print(f"VaR(5%): {stats['VaR_5%']:.4f}, CVaR(5%): {stats['CVaR_5%']:.4f}")
        print("Distribution of PnL:", stats['distribution'])
    else:
        bench, slip, fill = None, None, None

    if args.mode == 'eval_mc':
        plt.hist(stats['distribution'], bins=30)
        plt.title('Distribution of PnL from Monte Carlo Simulations')
        plt.xlabel('PnL')
        plt.ylabel('Frequency')
        plt.show()
    elif args.mode == 'train':
        if hasattr(agent, 'losses'):
            plt.plot(agent.losses)
            plt.title('DQN Training Loss over Episodes')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.show()

if __name__ == "__main__":
    main()