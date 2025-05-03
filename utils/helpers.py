import torch
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def polyak_averaging(scores, alpha=0.99):
    smoothed_scores = []
    smoothed_value = scores[0]  # Initialize with the first score
    for score in scores:
        smoothed_value = alpha * smoothed_value + (1 - alpha) * score
        smoothed_scores.append(smoothed_value)
    return smoothed_scores