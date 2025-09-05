import subprocess
import os
import re
import numpy as np
import matplotlib.pyplot as plt

def run_eval_mc_for_tickers_with_plots(tickers, model_path, sims=200, sigma=0.01, target_qty=10):
    results = {}
    for ticker in tickers:
        data_file = f"data/{ticker}_real_tick_data.csv"
        cmd = [
            "python3", "main.py",
            "--mode", "eval_mc",
            "--data", data_file,
            "--model", model_path,
            "--sims", str(sims),
            "--sigma", str(sigma),
            "--target_qty", str(target_qty)
        ]
        print(f"Running eval_mc for {ticker}...")
        try:
            completed_process = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = completed_process.stdout
            print(output)

            # Parse the distribution from output
            distribution = parse_distribution(output)
            if distribution is not None:
                # Create folder
                folder = f"results/{ticker}"
                os.makedirs(folder, exist_ok=True)

                # Plot histogram
                plt.figure(figsize=(10, 6))
                plt.hist(distribution, bins=30, alpha=0.7, color='blue', edgecolor='black')
                plt.title(f'Distribution of PnL for {ticker}')
                plt.xlabel('PnL')
                plt.ylabel('Frequency')
                plt.grid(True)

                # Save plot
                plot_path = os.path.join(folder, f"{ticker}_pnl_distribution.png")
                plt.savefig(plot_path)
                plt.close()
                print(f"Plot saved to {plot_path}")

            results[ticker] = output
        except subprocess.CalledProcessError as e:
            print(f"Error running eval_mc for {ticker}: {e.stderr}")
            results[ticker] = None
    return results

def parse_distribution(output):
    # Find the line with "Distribution of PnL:"
    lines = output.split('\n')
    for i, line in enumerate(lines):
        if "Distribution of PnL:" in line:
            # Next line should have the array
            if i + 1 < len(lines):
                array_str = lines[i + 1].strip()
                # Remove brackets and split
                array_str = array_str.replace('[', '').replace(']', '')
                values = array_str.split()
                try:
                    distribution = np.array([float(v) for v in values])
                    return distribution
                except ValueError:
                    print("Error parsing distribution values")
                    return None
    return None

if __name__ == "__main__":
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META']
    model_path = "models/DQN/double_dqn.pth"
    run_eval_mc_for_tickers_with_plots(tickers, model_path)
