import numpy as np
from scipy.stats import norm

def vwap(prices, volumes):
    return np.sum(prices * volumes) / np.sum(volumes)

def slippage(executed, benchmark):
    return np.mean(executed - benchmark)

def fill_ratio(total_executed, target):
    return total_executed / target

def latency_penalty(latencies, cost_per_ms=0.1):
    return np.mean(latencies) * cost_per_ms

def var_cvar(pnl_series, alpha=0.05):
    pnl = np.sort(pnl_series)
    idx = int(alpha * len(pnl))
    var = pnl[idx]
    cvar = pnl[:idx].mean() if idx>0 else var
    return var, cvar

def monte_carlo_scenarios(tick_data, sims=1000, sigma=0.005):
    """
    Generate MonteCarlo paths by adding Gaussian noise to mid_price
    """
    base = tick_data.copy()
    scenarios = []
    for _ in range(sims):
        noise = np.random.normal(0, sigma, size=base.shape)
        scenarios.append(base * (1 + noise))
    return scenarios
