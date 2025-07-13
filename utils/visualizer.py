import matplotlib.pyplot as plt
import seaborn as sns

def plot_execution(price_series, exec_pts, title="Execution Path"):
    plt.figure(figsize=(12,4))
    plt.plot(price_series, label='Mid Price', lw=1)
    buys  = [(i,p) for i,(p,a) in enumerate(exec_pts) if a>0]
    sells = [(i,p) for i,(p,a) in enumerate(exec_pts) if a<0]
    if buys:
        ix, py = zip(*buys)
        plt.scatter(ix, py, marker='^', color='g', label='Buys')
    if sells:
        ix, py = zip(*sells)
        plt.scatter(ix, py, marker='v', color='r', label='Sells')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pnl_distribution(pnl_series, var, cvar):
    sns.histplot(pnl_series, bins=50, kde=True)
    plt.axvline(var, color='r', linestyle='--', label=f'VaR (5%): {var:.4f}')
    plt.axvline(cvar, color='m', linestyle='-.', label=f'CVaR: {cvar:.4f}')
    plt.legend()
    plt.title("PnL Distribution")
    plt.show()
