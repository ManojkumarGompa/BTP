import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import ale_py
from agent.multi_trajectory_sfac_agent import MultiTrajectorySFACAgent
from utils import config as cfg
import os

# Register Atari environments
gym.register_envs(ale_py)

# Environment and training parameters
ENV_NAME = 'ALE/Assault-v5'
NUM_EPISODES = 1000
MAX_STEPS = 10000
SAVE_INTERVAL = 50  # Save model every 50 episodes
LOG_INTERVAL = 10   # Log progress every 10 episodes
RENDER_INTERVAL = 100  # Render every 100 episodes
NUM_TRAJECTORIES = 5  # Number of trajectories to collect before updating

# Create directories for saving models and results
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def train_agent():
    # Create environment
    env = gym.make(ENV_NAME, render_mode='rgb_array')
    
    # Get state and action dimensions
    state, _ = env.reset()
    state_dim = state.shape
    action_dim = env.action_space.n
    
    print(f"State dimensions: {state_dim}")
    print(f"Action dimensions: {action_dim}")
    
    # Initialize agent
    agent = MultiTrajectorySFACAgent(state_dim, action_dim)
    
    # Training metrics
    episode_rewards = []
    average_rewards = []
    beta_values = []
    perturbation_values = []
    
    total_trajectories = 0
    episode = 0
    
    # Reset on-policy buffers at start
    agent.reset_buffers()
    
    # Training loop - now collects multiple trajectories before updates
    while episode < NUM_EPISODES:
        # Run a trajectory
        state, _ = env.reset()
        trajectory_reward = 0
        
        for step in range(MAX_STEPS):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
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
            print(f"Updated policy after {NUM_TRAJECTORIES} trajectories")
            print(f"Average trajectory reward: {sum(agent.trajectory_rewards) / NUM_TRAJECTORIES:.2f}")
            print(f"Beta = {agent.beta:.4f}, Perturbations = {agent.T}")
            
            # Update tracking metrics
            episode += 1
            episode_reward = sum(agent.trajectory_rewards) / NUM_TRAJECTORIES
            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards[-100:])  # Moving average over last 100 episodes
            average_rewards.append(avg_reward)
            beta_values.append(agent.beta)
            perturbation_values.append(agent.T)
            
            # More detailed logging every LOG_INTERVAL episodes
            if episode % LOG_INTERVAL == 0:
                print(f"Episode {episode}: Reward = {episode_reward:.2f}, Avg Reward = {avg_reward:.2f}")
            
            # Save model periodically
            if episode % SAVE_INTERVAL == 0:
                torch.save(agent.actor.state_dict(), f'models/actor_episode_{episode}.pth')
                torch.save(agent.critic.state_dict(), f'models/critic_episode_{episode}.pth')
                
                # Plot and save training curves
                plt.figure(figsize=(15, 10))
                
                plt.subplot(2, 2, 1)
                plt.plot(episode_rewards)
                plt.title('Episode Rewards (Avg per Trajectory Batch)')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                
                plt.subplot(2, 2, 2)
                plt.plot(average_rewards)
                plt.title('Average Rewards (last 100 episodes)')
                plt.xlabel('Episode')
                plt.ylabel('Average Reward')
                
                plt.subplot(2, 2, 3)
                plt.plot(beta_values)
                plt.title('Beta Values (Smoothing Factor)')
                plt.xlabel('Episode')
                plt.ylabel('Beta')
                
                plt.subplot(2, 2, 4)
                plt.plot(perturbation_values)
                plt.title('Number of Perturbations (T)')
                plt.xlabel('Episode')
                plt.ylabel('Perturbations')
                
                plt.tight_layout()
                plt.savefig(f'results/training_curves_episode_{episode}.png')
                plt.close()
                
            # Render episode for visualization
            if episode % RENDER_INTERVAL == 0:
                render_episode(agent, env, episode)
    
    # Save final model
    torch.save(agent.actor.state_dict(), 'models/actor_final.pth')
    torch.save(agent.critic.state_dict(), 'models/critic_final.pth')
    
    # Close environment
    env.close()
    
    return episode_rewards, average_rewards, beta_values, perturbation_values

def render_episode(agent, env, episode_num):
    """Render a single episode with the current policy"""
    env_render = gym.make(ENV_NAME, render_mode='rgb_array')
    state, _ = env_render.reset()
    frames = []
    
    done = False
    episode_reward = 0
    
    while not done:
        # Select action
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = agent.actor(state_tensor)
        action = agent.actor.sample_action(action_probs[0])
        
        # Take action
        next_state, reward, done, truncated, _ = env_render.step(action)
        episode_reward += reward
        
        # Collect frame for visualization
        frame = env_render.render()
        frames.append(frame)
        
        # Update state
        state = next_state
        
        if truncated:
            done = True
    
    env_render.close()
    
    # Print episode reward
    print(f"Rendered episode {episode_num} - Reward: {episode_reward:.2f}")

def plot_final_results(rewards, avg_rewards, betas, perturbations):
    """Plot final training results"""
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards (Avg per Trajectory Batch)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(2, 2, 2)
    plt.plot(avg_rewards)
    plt.title('Average Rewards (last 100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    plt.subplot(2, 2, 3)
    plt.plot(betas)
    plt.title('Beta Values (Smoothing Factor)')
    plt.xlabel('Episode')
    plt.ylabel('Beta')
    
    plt.subplot(2, 2, 4)
    plt.plot(perturbations)
    plt.title('Number of Perturbations (T)')
    plt.xlabel('Episode')
    plt.ylabel('Perturbations')
    
    plt.tight_layout()
    plt.savefig('results/final_training_curves.png')
    plt.show()

if __name__ == "__main__":
    rewards, avg_rewards, betas, perturbations = train_agent()
    plot_final_results(rewards, avg_rewards, betas, perturbations)
    
    print("Training completed!")
    print(f"Final average reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")