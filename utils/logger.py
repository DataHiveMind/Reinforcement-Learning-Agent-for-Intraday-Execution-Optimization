import os

class DummyLogger:
    def add_scalar(self, tag, value, step):
        print(f"{tag}: {value} at step {step}")

def get_logger(log_dir='runs/exp'):
    os.makedirs(log_dir, exist_ok=True)
    return DummyLogger()

def log_metrics(writer, ep, reward, avg_latency=None):
    writer.add_scalar('Episode/Reward', reward, ep)
    if avg_latency is not None:
        writer.add_scalar('Episode/AvgLatency', avg_latency, ep)
