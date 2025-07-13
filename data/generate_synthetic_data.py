import numpy as np
import pandas as pd

def simulate_tick_data(
    initial_price=100.0,
    steps=1000,
    drift=0.0001,
    vol=0.001,
    base_spread=0.01,
    spread_vol=0.002,
    imb_noise=0.1
):
    """
    Simulate tick-level data with mid-price following a drift+noise process,
    spread varying around a base level, and imbalance noise.
    """
    mid_prices = [initial_price]
    spreads    = []
    imbalances = []

    for t in range(1, steps):
        # Geometric Brownian Motion step
        prev = mid_prices[-1]
        shock = np.random.normal(drift, vol)
        mid = prev * (1 + shock)
        mid_prices.append(mid)

        # spread ~ base_spread + noise
        spreads.append(abs(np.random.normal(base_spread, spread_vol)))
        # imbalance ~ Uniform[-1,1] + small noise
        imbalances.append(np.tanh(np.random.normal(0, imb_noise)))

    # first spread/imb=repeat
    spreads.insert(0, abs(np.random.normal(base_spread, spread_vol)))
    imbalances.insert(0, np.tanh(np.random.normal(0, imb_noise)))

    df = pd.DataFrame({
        'mid_price': mid_prices,
        'spread': spreads,
        'imbalance': imbalances
    })
    return df

if __name__ == "__main__":
    df = simulate_tick_data(steps=1000)
    df.to_csv('data/mock_tick_data.csv', index=False)
    print("Generated synthetic tick data: data/mock_tick_data.csv")
