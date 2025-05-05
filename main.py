import gymnasium as gym
import numpy as np
import torch
import os
import time
# from torch.utils.tensorboard import Summarywriter
from datetime import datetime

from agents.actor import Actor
from agents.critic import Critic
from agents.sfac_agent import MultiTrajectorySFACAgent
from utils.preprocessing import AtariPreprocessor
from utils import config as cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def make_env(env_name,render_mode=None):
    """Create and wrap the Atari environment"""
    env = gym.make(env_name, render_mode=render_mode)
    env.reset()
    return env

import matplotlib.pyplot as plt
import json
from utils.plotting import plot_learning_curve, plot_comparison, save_rewards_to_file

def train_sfac():
    """Train the SFAC agent on the specified environment"""
    env_name_safe = cfg.ENV_NAME.replace('/', '_')
    # Create log directory
    run_name = f"SFAC_{env_name_safe}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join("logs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Setup tensorboard
    #writer = SummaryWriter(log_dir)
    
    # Create environment
    env = make_env(cfg.ENV_NAME, render_mode=None)
    eval_env = make_env(cfg.ENV_NAME, render_mode='rgb_array' if cfg.RENDER_EVAL else None)
    
    # Setup preprocessor
    preprocessor = AtariPreprocessor(frame_stack=cfg.FRAME_STACK, frame_size=cfg.FRAME_SIZE, device=device)
    
    # Get initial state to determine dimensions
    state, _ = env.reset()
    processed_state = preprocessor.process_state(state, reset=True)
    state_dim = processed_state.shape  # Should be (4, 84, 84) for stacked frames
    action_size = env.action_space.n
    
    print(f"State shape: {state_dim}, Action size: {action_size}")
    
    # Initialize networks with improved architecture
    actor = Actor(state_dim=state_dim, action_size=action_size, 
                 hidden1_dim=cfg.HIDDEN1_DIM, hidden2_dim=cfg.HIDDEN2_DIM).to(device)
    critic = Critic(state_dim=state_dim, action_size=action_size,
                   hidden1_dim=cfg.HIDDEN1_DIM, hidden2_dim=cfg.HIDDEN2_DIM).to(device)
    
    # Initialize agent
    agent = MultiTrajectorySFACAgent(state_dim=state_dim, action_size=action_size, 
                                    actor=actor, critic=critic)
    
    # Training variables
    total_steps = 0
    best_eval_reward = float('-inf')
    no_improvement_count = 0
    episode = 0
    episode_rewards = []  # Track all episode rewards
    
    # Main training loop
    while episode < cfg.NUM_EPISODES:
        episode += 1
        state, _ = env.reset()
        state = preprocessor.process_state(state, reset=True)
        episode_reward = 0
        episode_steps = 0
        done = False
        
        # Collect trajectory
        while not done and episode_steps < cfg.MAX_STEPS_PER_EPISODE:
            # Convert state to tensor and get action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action_probs = agent.actor(state_tensor)
            action = agent.actor.sample_action(action_probs[0])
            
            # Take step in environment
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = preprocessor.process_state(next_state)
            
            # Clip reward for stability
            clipped_reward = preprocessor.clip_reward(reward)
            
            # Store transition
            agent.store_transition(state, action, clipped_reward, next_state, done or truncated)
            
            # Update tracking variables
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            # Update ONLY when we've collected enough transitions
            if len(agent.states) >= cfg.UPDATE_EVERY:
                print(f"Updating agent at step {total_steps}")
                agent.update(batch_size=cfg.BATCH_SIZE)
                
            # If done, start a new episode
            if done or truncated:
                break
        
        # Store episode reward
        agent.store_episode_reward(episode_reward)
        episode_rewards.append(episode_reward)  # Add to our tracking list
        
        # Log episode information
        agent.adjust_beta()
        print(f"Episode {episode}: Reward = {episode_reward:.2f}, Steps = {episode_steps}")
        #writer.add_scalar("Training/Episode_Reward", episode_reward, episode)
        # Writer.add_scalar("Training/Episode_Length", episode_steps, episode)
        # writer.add_scalar("Parameters/Beta", agent.beta, episode)
        #writer.add_scalar("Parameters/T", agent.T, episode)
        
        # Periodic saving and plotting
        if episode % cfg.SAVE_FREQUENCY == 0:
            # Save checkpoint
            checkpoint_path = os.path.join("checkpoints", f"{run_name}_ep{episode}.pt")
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'episode': episode
            }, checkpoint_path)
            
            # Create and save plot
            plot_path = os.path.join("plots", f"{run_name}_ep{episode}.png")
            plot_learning_curve(episode_rewards, "SFAC", save_path=plot_path)
            print(f"Plot saved to {plot_path}")
            
            # Save rewards data
            rewards_path = save_rewards_to_file(episode_rewards, "SFAC", run_name)
            print(f"Rewards data saved to {rewards_path}")
            
            # Memory cleanup
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    # Save final model
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
        'episode': episode
    }, os.path.join("checkpoints", f"{run_name}_final.pt"))
    
    # Save final plot and rewards data
    final_plot_path = os.path.join("plots", f"{run_name}_final.png")
    plot_learning_curve(episode_rewards, "SFAC", save_path=final_plot_path)
    final_rewards_path = save_rewards_to_file(episode_rewards, "SFAC", run_name)
    
    #writer.close()
    env.close()
    eval_env.close()
    
    return agent, episode_rewards, run_name

def train_policy_gradient():
    """Train a regular policy gradient agent for comparison"""
    from agents.policy_gradient_agent import PolicyGradientAgent
    
    # Create sanitized environment name for file paths
    env_name_safe = cfg.ENV_NAME.replace('/', '_')
    
    # Create log directory
    run_name = f"PG_{env_name_safe}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join("logs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Setup tensorboard
    #writer = SummaryWriter(log_dir)
    
    # Create environment
    env = make_env(cfg.ENV_NAME, render_mode=None)
    eval_env = make_env(cfg.ENV_NAME, render_mode='rgb_array' if cfg.RENDER_EVAL else None)
    
    # Setup preprocessor
    preprocessor = AtariPreprocessor(frame_stack=cfg.FRAME_STACK, frame_size=cfg.FRAME_SIZE, device=device)
    
    # Get initial state to determine dimensions
    state, _ = env.reset()
    processed_state = preprocessor.process_state(state, reset=True)
    state_dim = processed_state.shape
    action_size = env.action_space.n
    
    print(f"State shape: {state_dim}, Action size: {action_size}")
    
    # Initialize networks
    actor = Actor(state_dim=state_dim, action_size=action_size, 
                 hidden1_dim=cfg.HIDDEN1_DIM, hidden2_dim=cfg.HIDDEN2_DIM).to(device)
    critic = Critic(state_dim=state_dim, action_size=action_size,
                   hidden1_dim=cfg.HIDDEN1_DIM, hidden2_dim=cfg.HIDDEN2_DIM).to(device)
    
    # Initialize policy gradient agent
    agent = PolicyGradientAgent(state_dim=state_dim, action_size=action_size, 
                               actor=actor, critic=critic)
    
    # Training variables
    total_steps = 0
    best_eval_reward = float('-inf')
    no_improvement_count = 0
    episode = 0
    episode_rewards = []  # Track all episode rewards
    
    # Main training loop
    while episode < cfg.NUM_EPISODES:
        episode += 1
        state, _ = env.reset()
        state = preprocessor.process_state(state, reset=True)
        episode_reward = 0
        episode_steps = 0
        done = False
        
        # Collect trajectory
        while not done and episode_steps < cfg.MAX_STEPS_PER_EPISODE:
            # Convert state to tensor and get action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action_probs = agent.actor(state_tensor)
            action = agent.actor.sample_action(action_probs[0])
            
            # Take step in environment
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = preprocessor.process_state(next_state)
            
            # Clip reward for stability
            clipped_reward = preprocessor.clip_reward(reward)
            
            # Store transition
            agent.store_transition(state, action, clipped_reward, next_state, done or truncated)
            
            # Update tracking variables
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            # Update ONLY when we've collected enough transitions
            if len(agent.states) >= cfg.UPDATE_EVERY:
                print(f"Updating agent at step {total_steps}")
                losses = agent.update(batch_size=cfg.BATCH_SIZE)
                if losses:
                    critic_loss, actor_loss = losses
                    #writer.add_scalar("Training/Critic_Loss", critic_loss, total_steps)
                    #writer.add_scalar("Training/Actor_Loss", actor_loss, total_steps)
                
            # If done, start a new episode
            if done or truncated:
                break
        
        # Store episode reward
        agent.store_episode_reward(episode_reward)
        episode_rewards.append(episode_reward)  # Add to our tracking list
        
        # Log episode information
        print(f"Episode {episode}: Reward = {episode_reward:.2f}, Steps = {episode_steps}")
        #writer.add_scalar("Training/Episode_Reward", episode_reward, episode)
        #writer.add_scalar("Training/Episode_Length", episode_steps, episode)
        
        # Periodic saving and plotting
        if episode % cfg.SAVE_FREQUENCY == 0:
            # Save checkpoint
            checkpoint_path = os.path.join("checkpoints", f"{run_name}_ep{episode}.pt")
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'episode': episode
            }, checkpoint_path)
            
            # Create and save plot
            plot_path = os.path.join("plots", f"{run_name}_ep{episode}.png")
            plot_learning_curve(episode_rewards, "Policy Gradient", save_path=plot_path)
            print(f"Plot saved to {plot_path}")
            
            # Save rewards data
            rewards_path = save_rewards_to_file(episode_rewards, "PG", run_name)
            print(f"Rewards data saved to {rewards_path}")
            
            # Memory cleanup
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    # Save final model
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
        'episode': episode
    }, os.path.join("checkpoints", f"{run_name}_final.pt"))
    
    # Save final plot and rewards data
    final_plot_path = os.path.join("plots", f"{run_name}_final.png")
    plot_learning_curve(episode_rewards, "Policy Gradient", save_path=final_plot_path)
    final_rewards_path = save_rewards_to_file(episode_rewards, "PG", run_name)
    
    #writer.close()
    env.close()
    eval_env.close()
    
    return agent, episode_rewards, run_name

def evaluate(agent, env, preprocessor, num_episodes=cfg.EVAL_EPISODES, render=cfg.RENDER_EVAL):
    """Evaluate agent performance"""
    total_rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        state = preprocessor.process_state(state, reset=True)
        episode_reward = 0
        done = False
        
        while not done:
            # Render if enabled
            if render:
                env.render()
                
            # Get action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action_probs = agent.actor(state_tensor)
                
            # Use greedy action for evaluation (no exploration)
            action = torch.argmax(action_probs).item()
            
            # Take step
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = preprocessor.process_state(next_state)
            
            # Update
            state = next_state
            episode_reward += reward
            if truncated:
                done = True
                
        total_rewards.append(episode_reward)
        
    avg_reward = np.mean(total_rewards)
    print(f"Evaluation: Average Reward = {avg_reward:.2f} over {num_episodes} episodes")
    return avg_reward

def main():
    # Set seeds
    # torch.manual_seed(cfg.SEED)
    # np.random.seed(cfg.SEED)
    
    # Train based on selected algorithm in config
    if cfg.ALGORITHM == "sfac":
        pg_agent,pg_rewards,pg_run_name = train_policy_gradient()
        sfac_agent,sfac_rewards,sfac_run_name = train_sfac()
        comparison_path = os.path.join("plots", f"comparison_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png")
        plot_comparison(sfac_rewards=sfac_rewards, pg_rewards=pg_rewards, save_path=comparison_path)
        print(f"Comparison plot saved to {comparison_path}")
    
    
    print("Training completed!")

if __name__ == "__main__":
    main()