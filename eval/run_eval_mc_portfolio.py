import subprocess

def run_eval_mc_for_tickers(tickers, model_path, sims=200, sigma=0.01, target_qty=10):
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
            results[ticker] = output
        except subprocess.CalledProcessError as e:
            print(f"Error running eval_mc for {ticker}: {e.stderr}")
            results[ticker] = None
    return results

if __name__ == "__main__":
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META']
    model_path = "models/DQN/double_dqn.pth"
    run_eval_mc_for_tickers(tickers, model_path)
