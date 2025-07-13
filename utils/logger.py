import os
from torch.utils.tensorboard import SummaryWriter

def get_logger(log_dir='runs/exp'):
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)

def log_metrics(writer, ep, reward, avg_latency=None):
    writer.add_scalar('Episode/Reward', reward, ep)
    if avg_latency is not None:
        writer.add_scalar('Episode/AvgLatency', avg_latency, ep)
