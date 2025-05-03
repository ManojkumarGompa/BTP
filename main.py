import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import ale_py
import os
import argparse
from agents.sfac_agent import MultiTrajectorySFACAgent
from agents.policy_gradient_agent import PolicyGradientAgent
from utils import config as cfg

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Register Atari environments
gym.register_envs(ale_py)

def parse_args():
    parser = argparse.ArgumentParser(description='Train and compare SFAC with Regular Policy Gradient')
    parser.add_argument('--env', type=str, default='ALE/Assault-v5', help='Atari environment name')
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes to train')
    parser.add_argument('--perturbations', type=int, default=500, help='Initial number of perturbations for SFAC')
    parser.add_argument('--trajectories', type=int, default=5, help='Number of trajectories per update')
    parser.add_argument('--mode', choices=['sfac', 'pg', 'both'], default='both', 
                        help='Which algorithm to run: sfac, pg, or both')
    return parser.parse_args()

def run_comparison():
    # Parse command-line arguments
    args = parse_args()
    
    ENV_NAME = args.env
    NUM_EPISODES = args.episodes
    NUM_TRAJECTORIES = args.trajectories
    MAX_STEPS = 10000
    SAVE_INTERVAL = 50
    LOG_INTERVAL = 10
    
    # Create directories for saving models and results
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Create environment
    env = gym.make(ENV_NAME, render_mode='rgb_array')
    
    # Get state and action dimensions
    state, _ = env.reset()
    state_dim = state.shape
    action_dim = env.action_space.n
    
    print(f"Environment: {ENV_NAME}")
    print(f"State dimensions: {state_dim}")
    print(f"Action dimensions: {action_dim}")
    print(f"Training for {NUM_EPISODES} episodes")
    print(f"Using {NUM_TRAJECTORIES} trajectories per update")
    
    # Initialize results
    sfac_rewards = []
    pg_rewards = []
    sfac_avg_rewards = []
    pg_avg_rewards = []
    beta_values = []
    perturbation_values = []
    
    # Run SFAC
    if args.mode in ['sfac', 'both']:
        print("\n=== Training SFAC Agent ===")
        sfac_agent = MultiTrajectorySFACAgent(state_dim, action_dim)
        sfac_agent.T = args.perturbations  # Set initial perturbation count
        
        sfac_rewards, sfac_avg_rewards, beta_values, perturbation_values = train_sfac(
            env, sfac_agent, NUM_EPISODES, MAX_STEPS, NUM_TRAJECTORIES, SAVE_INTERVAL, LOG_INTERVAL
        )
    
    # Run Policy Gradient
    if args.mode in ['pg', 'both']:
        print("\n=== Training Policy Gradient Agent ===")
        pg_agent = PolicyGradientAgent(state_dim, action_dim)
        
        pg_rewards, pg_avg_rewards = train_policy_gradient(
            env, pg_agent, NUM_EPISODES, MAX_STEPS, NUM_TRAJECTORIES, SAVE_INTERVAL, LOG_INTERVAL
        )
    
    # Plot comparison if both algorithms were run
    if args.mode == 'both':
        plot_comparison(
            sfac_rewards, pg_rewards, 
            sfac_avg_rewards, pg_avg_rewards,
            beta_values, perturbation_values,
            ENV_NAME, args.perturbations
        )
    
    env.close()
    
    print("\nTraining completed!")
    
    if args.mode in ['sfac', 'both']:
        print(f"SFAC Final Avg Reward (last 100 episodes): {np.mean(sfac_rewards[-100:]):.2f}")
    if args.mode in ['pg', 'both']:
        print(f"PG Final Avg Reward (last 100 episodes): {np.mean(pg_rewards[-100:]):.2f}")

def train_sfac(env, agent, num_episodes, max_steps, num_trajectories, save_interval, log_interval):
    """Train the SFAC agent and return metrics"""
    episode_rewards = []
    average_rewards = []
    beta_values = []
    perturbation_values = []
    
    total_trajectories = 0
    episode = 0
    
    # Reset on-policy buffers at start
    agent.reset_buffers()
    
    # Training loop
    while episode < num_episodes:
        # Run a trajectory
        state, _ = env.reset()
        trajectory_reward = 0
        
        for step in range(max_steps):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Select action
            action_probs = agent.actor(state_tensor)
            action = agent.actor.sample_action(action_probs[0])
            
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store transition in on-policy buffer
            agent.store_transition(state, action, reward, next_state, done or truncated)
            
            # Update state and metrics
            state = next_state
            trajectory_reward += reward
            
            if done or truncated:
                break
        
        # Store trajectory reward
        agent.store_trajectory_reward(trajectory_reward)
        total_trajectories += 1
        
        # Update policy after collecting NUM_TRAJECTORIES trajectories
        if agent.has_enough_trajectories():
            agent.update()
            
            # Log progress
            print(f"SFAC Updated policy after {num_trajectories} trajectories")
            print(f"Average trajectory reward: {sum(agent.trajectory_rewards) / num_trajectories:.2f}")
            print(f"Beta = {agent.beta:.4f}, Perturbations = {agent.T}")
            
            # Update tracking metrics
            episode += 1
            episode_reward = sum(agent.trajectory_rewards) / num_trajectories
            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            average_rewards.append(avg_reward)
            beta_values.append(agent.beta)
            perturbation_values.append(agent.T)
            
            # More detailed logging every LOG_INTERVAL episodes
            if episode % log_interval == 0:
                print(f"SFAC Episode {episode}: Reward = {episode_reward:.2f}, Avg Reward = {avg_reward:.2f}")
            
            # Save model periodically
            if episode % save_interval == 0:
                torch.save(agent.actor.state_dict(), f'models/sfac_actor_episode_{episode}.pth')
                torch.save(agent.critic.state_dict(), f'models/sfac_critic_episode_{episode}.pth')
                
                # Save intermediate plot
                save_intermediate_plot(
                    episode_rewards, average_rewards, 
                    beta_values, perturbation_values,
                    'SFAC', episode
                )
            
            agent.reset_buffers()
    
    # Save final model
    torch.save(agent.actor.state_dict(), 'models/sfac_actor_final.pth')
    torch.save(agent.critic.state_dict(), 'models/sfac_critic_final.pth')
    
    return episode_rewards, average_rewards, beta_values, perturbation_values

def train_policy_gradient(env, agent, num_episodes, max_steps, num_trajectories, save_interval, log_interval):
    """Train the Policy Gradient agent and return metrics"""
    episode_rewards = []
    average_rewards = []
    
    total_trajectories = 0
    episode = 0
    
    # Reset on-policy buffers at start
    agent.reset_buffers()
    
    # Training loop
    while episode < num_episodes:
        # Run a trajectory
        state, _ = env.reset()
        trajectory_reward = 0
        
        for step in range(max_steps):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Select action
            action_probs = agent.actor(state_tensor)
            action = agent.actor.sample_action(action_probs[0])
            
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store transition in on-policy buffer
            agent.store_transition(state, action, reward, next_state, done or truncated)
            
            # Update state and metrics
            state = next_state
            trajectory_reward += reward
            
            if done or truncated:
                break
        
        # Store trajectory reward
        agent.store_trajectory_reward(trajectory_reward)
        total_trajectories += 1
        
        # Update policy after collecting NUM_TRAJECTORIES trajectories
        if agent.has_enough_trajectories():
            agent.update()
            
            # Log progress
            print(f"PG Updated policy after {num_trajectories} trajectories")
            print(f"Average trajectory reward: {sum(agent.trajectory_rewards) / num_trajectories:.2f}")
            
            # Update tracking metrics
            episode += 1
            episode_reward = sum(agent.trajectory_rewards) / num_trajectories
            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            average_rewards.append(avg_reward)
            
            # More detailed logging every LOG_INTERVAL episodes
            if episode % log_interval == 0:
                print(f"PG Episode {episode}: Reward = {episode_reward:.2f}, Avg Reward = {avg_reward:.2f}")
            
            # Save model periodically
            if episode % save_interval == 0:
                torch.save(agent.actor.state_dict(), f'models/pg_actor_episode_{episode}.pth')
                torch.save(agent.critic.state_dict(), f'models/pg_critic_episode_{episode}.pth')
                
                # Save intermediate plot
                save_intermediate_plot(
                    episode_rewards, average_rewards, 
                    None, None,
                    'Policy Gradient', episode
                )
            
            agent.reset_buffers()
    
    # Save final model
    torch.save(agent.actor.state_dict(), 'models/pg_actor_final.pth')
    torch.save(agent.critic.state_dict(), 'models/pg_critic_final.pth')
    
    return episode_rewards, average_rewards

def save_intermediate_plot(rewards, avg_rewards, betas=None, perturbations=None, algorithm='SFAC', episode=0):
    """Save intermediate plot during training"""
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(rewards, 'b-', alpha=0.3)
    plt.plot(avg_rewards, 'b-', linewidth=2)
    plt.title(f'{algorithm} Rewards - Episode {episode}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    
    # Plot average rewards
    plt.subplot(2, 2, 2)
    plt.plot(avg_rewards)
    plt.title(f'{algorithm} Average Rewards (last 100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True, alpha=0.3)
    
    if betas is not None:
        # Plot beta values
        plt.subplot(2, 2, 3)
        plt.plot(betas)
        plt.title('Beta Values (Smoothing Factor)')
        plt.xlabel('Episode')
        plt.ylabel('Beta')
        plt.grid(True, alpha=0.3)
    
    if perturbations is not None:
        # Plot perturbation values
        plt.subplot(2, 2, 4)
        plt.plot(perturbations)
        plt.title('Number of Perturbations (T)')
        plt.xlabel('Episode')
        plt.ylabel('Perturbations')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/{algorithm.lower()}_training_curves_episode_{episode}.png')
    plt.close()

def plot_comparison(sfac_rewards, pg_rewards, sfac_avg, pg_avg, betas, perturbations, env_name, perturb_count):
    """Plot comparison between SFAC and Policy Gradient"""
    game_name = env_name.split('/')[-1]
    
    plt.figure(figsize=(15, 10))
    
    # Plot raw rewards
    plt.subplot(2, 2, 1)
    plt.plot(sfac_rewards, 'b-', alpha=0.3, label='SFAC (raw)')
    plt.plot(pg_rewards, 'r-', alpha=0.3, label='PG (raw)')
    plt.plot(sfac_avg, 'b-', linewidth=2, label='SFAC (avg)')
    plt.plot(pg_avg, 'r-', linewidth=2, label='PG (avg)')
    plt.title(f'Rewards Comparison - {game_name}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot cumulative rewards
    plt.subplot(2, 2, 2)
    plt.plot(np.cumsum(sfac_rewards), 'b-', label='SFAC')
    plt.plot(np.cumsum(pg_rewards), 'r-', label='Policy Gradient')
    plt.title('Cumulative Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot beta values
    plt.subplot(2, 2, 3)
    plt.plot(betas)
    plt.title('Beta Values (Smoothing Factor)')
    plt.xlabel('Episode')
    plt.ylabel('Beta')
    plt.grid(True, alpha=0.3)
    
    # Plot perturbation values
    plt.subplot(2, 2, 4)
    plt.plot(perturbations)
    plt.title('Number of Perturbations (T)')
    plt.xlabel('Episode')
    plt.ylabel('Perturbations')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/comparison_{game_name}_{perturb_count}.png')
    plt.show()
    
    # Save data for later analysis
    np.savez(
        f'results/comparison_data_{game_name}_{perturb_count}.npz',
        sfac_rewards=sfac_rewards,
        pg_rewards=pg_rewards,
        sfac_avg=sfac_avg,
        pg_avg=pg_avg,
        betas=betas,
        perturbations=perturbations
    )
    
    # Also create a detailed performance comparison plot
    plt.figure(figsize=(18, 10))
    
    # Create episodes array
    episodes = np.arange(len(sfac_rewards))
    
    # Plot reward difference (SFAC - PG)
    plt.subplot(2, 1, 1)
    reward_diff = np.array(sfac_rewards) - np.array(pg_rewards)
    plt.plot(episodes, reward_diff)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.fill_between(episodes, reward_diff, 0, where=reward_diff > 0, facecolor='green', alpha=0.3, interpolate=True)
    plt.fill_between(episodes, reward_diff, 0, where=reward_diff < 0, facecolor='red', alpha=0.3, interpolate=True)
    plt.title(f'SFAC vs Policy Gradient Performance Difference - {game_name}')
    plt.xlabel('Episode')
    plt.ylabel('Reward Difference (SFAC - PG)')
    plt.grid(True, alpha=0.3)
    
    # Plot relative performance (SFAC/PG)
    plt.subplot(2, 1, 2)
    
    # Use rolling window for smoother visualization
    window_size = 10
    sfac_smooth = np.array(sfac_avg)
    pg_smooth = np.array(pg_avg)
    
    # Replace zeros with small value to avoid division issues
    pg_smooth = np.where(pg_smooth == 0, 0.0001, pg_smooth)
    
    relative_perf = sfac_smooth / pg_smooth
    plt.plot(episodes, relative_perf)
    plt.axhline(y=1, color='r', linestyle='-', alpha=0.3)
    plt.fill_between(episodes, relative_perf, 1, where=relative_perf > 1, facecolor='green', alpha=0.3, interpolate=True)
    plt.fill_between(episodes, relative_perf, 1, where=relative_perf < 1, facecolor='red', alpha=0.3, interpolate=True)
    plt.title('Relative Performance (SFAC/PG)')
    plt.xlabel('Episode')
    plt.ylabel('Performance Ratio')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/detailed_comparison_{game_name}_{perturb_count}.png')
    plt.close()

if __name__ == "__main__":
    run_comparison()