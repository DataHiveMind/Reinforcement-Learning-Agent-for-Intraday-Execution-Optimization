import numpy as np
from utils.metrics import vwap, slippage, fill_ratio, var_cvar, monte_carlo_scenarios

def single_run(agent, tick_data, target_qty=10):
    exec_prices, exec_qty = [], []
    env = agent.env if hasattr(agent,'env') else None
    # build a fresh env if needed
    from env.orderbook_env import OrderBookEnv
    env = OrderBookEnv(tick_data, max_steps=len(tick_data))
    state = env.reset()
    while True:
        act = agent.select_action(state)
        state, _, done, info = env.step(act)
        exec_prices.append(info['exec_cost'])
        exec_qty.append(info['inventory'])
        if done: break

    prices = np.array(exec_prices)
    vols   = np.diff([0]+exec_qty)
    bench  = vwap(prices, vols)
    slip   = slippage(prices, bench)
    fill   = fill_ratio(np.sum(vols), target_qty)
    return bench, slip, fill

def monte_carlo_backtest(agent, tick_data, sims=500, sigma=0.01, target_qty=10):
    scenarios = monte_carlo_scenarios(tick_data, sims, sigma)
    pnl_series = []
    for scenario in scenarios:
        bench, slip, fill = single_run(agent, scenario, target_qty)
        pnl = fill*(-bench)  # simplistic PnL estimate
        pnl_series.append(pnl)

    var, cvar = var_cvar(np.array(pnl_series), alpha=0.05)
    return {
        'pnl_mean' : np.mean(pnl_series),
        'pnl_std'  : np.std(pnl_series),
        'VaR_5%'   : var,
        'CVaR_5%'  : cvar,
        'distribution': np.array(pnl_series)
    }
