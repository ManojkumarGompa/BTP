import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import datetime

def polyak_moving_average(data, alpha=0.95):
    """
    Calculate Polyak (exponential) moving average
    
    Args:
        data: Input data array
        alpha: Smoothing factor (higher = smoother)
    
    Returns:
        Array of smoothed values
    """
    smoothed = np.zeros_like(data, dtype=float)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * data[i]
    return smoothed

def plot_learning_curve(rewards, algorithm_name, save_path=None, alpha=0.95):
    """
    Plot the learning curve using Polyak averaging
    
    Args:
        rewards: List of episode rewards
        algorithm_name: Name of the algorithm (SFAC or PG)
        save_path: Path to save the plot
        alpha: Polyak averaging coefficient
    """
    plt.figure(figsize=(12, 8))
    
    # Raw rewards
    episodes = np.arange(len(rewards)) + 1
    plt.plot(episodes, rewards, 'b-', alpha=0.3, label='Raw rewards')
    
    # Polyak-averaged rewards
    if len(rewards) > 1:
        smoothed = polyak_moving_average(rewards, alpha)
        plt.plot(episodes, smoothed, 'r-', linewidth=2, label=f'Polyak avg (α={alpha})')
    
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(f'Learning Curve - {algorithm_name}')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_comparison(sfac_rewards=None, pg_rewards=None, sfac_path=None, pg_path=None, save_path=None, alpha=0.95):
    """
    Plot a comparison between SFAC and PG algorithms using Polyak averaging
    
    Args:
        sfac_rewards: List of SFAC episode rewards (optional)
        pg_rewards: List of PG episode rewards (optional)
        sfac_path: Path to SFAC rewards JSON file (optional)
        pg_path: Path to PG rewards JSON file (optional)
        save_path: Path to save the comparison plot
        alpha: Polyak averaging coefficient
    """
    plt.figure(figsize=(12, 8))
    
    # Load rewards from files if not provided directly
    if sfac_rewards is None and sfac_path:
        with open(sfac_path, 'r') as f:
            sfac_data = json.load(f)
            sfac_rewards = sfac_data['rewards']
    
    if pg_rewards is None and pg_path:
        with open(pg_path, 'r') as f:
            pg_data = json.load(f)
            pg_rewards = pg_data['rewards']
    
    # Plot SFAC rewards if available
    if sfac_rewards:
        episodes_sfac = np.arange(len(sfac_rewards)) + 1
        if len(sfac_rewards) > 1:
            smoothed_sfac = polyak_moving_average(sfac_rewards, alpha)
            plt.plot(episodes_sfac, smoothed_sfac, 'b-', linewidth=2, label='SFAC')
        else:
            plt.plot(episodes_sfac, sfac_rewards, 'b-', label='SFAC')
    
    # Plot PG rewards if available
    if pg_rewards:
        episodes_pg = np.arange(len(pg_rewards)) + 1
        if len(pg_rewards) > 1:
            smoothed_pg = polyak_moving_average(pg_rewards, alpha)
            plt.plot(episodes_pg, smoothed_pg, 'r-', linewidth=2, label='Policy Gradient')
        else:
            plt.plot(episodes_pg, pg_rewards, 'r-', label='Policy Gradient')
    
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(f'SFAC vs Policy Gradient (Polyak α={alpha})')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def save_rewards_to_file(rewards, algorithm_name, run_name, save_dir="results"):
    """
    Save rewards to a JSON file
    
    Args:
        rewards: List of episode rewards
        algorithm_name: Name of the algorithm (SFAC or PG)
        run_name: Name of the run
        save_dir: Directory to save the file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    data = {
        'algorithm': algorithm_name,
        'run_name': run_name,
        'rewards': rewards,
        'timestamp': datetime.now().strftime('%Y%m%d-%H%M%S')
    }
    
    file_path = os.path.join(save_dir, f"{algorithm_name}_{run_name}_rewards.json")
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    return file_path