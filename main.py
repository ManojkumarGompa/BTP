import gymnasium as gym
import numpy as np
import torch
import os
import time
import json
import gc
import argparse
from datetime import datetime

from agents.actor import Actor
from agents.critic import Critic
from agents.sfac_agent import MultiTrajectorySFACAgent
from agents.policy_gradient_agent import PolicyGradientAgent
from utils.preprocessing import AtariPreprocessor
from utils import config as cfg
from utils.plotting import plot_beta_history, plot_learning_curve, plot_comparison, save_rewards_to_file

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def make_env(env_name, render_mode=None):
    """Create and wrap the Atari environment"""
    env = gym.make(env_name, render_mode=render_mode)
    env.reset()
    return env

def train_sfac(checkpoint_path=None):
    """Train the SFAC agent on the specified environment with checkpoint resumption"""
    env_name_safe = cfg.ENV_NAME.replace('/', '_')
    
    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
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
    
    # Initialize agent
    agent = MultiTrajectorySFACAgent(state_dim=state_dim, action_size=action_size, 
                                    actor=actor, critic=critic)
    
    # Initialize training variables
    start_episode = 0
    episode_rewards = []
    beta_history = []
    run_name = None
    
    # Load from checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # Load model weights
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.critic.load_state_dict(checkpoint['critic'])
        
        # Load optimizer states if available
        if 'actor_optimizer' in checkpoint:
            agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        if 'critic_optimizer' in checkpoint:
            agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            
        # Resume episode count
        start_episode = checkpoint['episode']
        print(f"Resuming from episode {start_episode + 1}")
        
        # Extract run name from checkpoint path if possible
        run_name_parts = os.path.basename(checkpoint_path).split('_ep')
        if len(run_name_parts) > 0:
            run_name = run_name_parts[0]
            print(f"Continuing run: {run_name}")
            
        # Try to load reward history
        rewards_file = checkpoint_path.replace(".pt", "_rewards.json")
        if os.path.exists(rewards_file):
            with open(rewards_file, 'r') as f:
                data = json.load(f)
                episode_rewards = data.get('rewards', [])
                print(f"Loaded {len(episode_rewards)} previous episode rewards")
                
        # Try to load beta history
        beta_file = checkpoint_path.replace(".pt", "_beta.json")
        if os.path.exists(beta_file):
            with open(beta_file, 'r') as f:
                data = json.load(f)
                beta_history = data.get('beta_history', [])
                if beta_history and len(beta_history) > 0:
                    agent.beta = beta_history[-1]  # Restore last beta value
                print(f"Loaded beta history with {len(beta_history)} entries")
                
        # Load T value if available
        if 'T' in checkpoint:
            agent.T = checkpoint['T']
            print(f"Loaded perturbation count T: {agent.T}")
            
    # Create a new run name if not resuming
    if run_name is None:
        run_name = f"SFAC_{env_name_safe}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        print(f"Starting new run: {run_name}")
    
    # Training variables
    total_steps = 0
    best_eval_reward = float('-inf')
    no_improvement_count = 0
    
    # Main training loop
    for episode in range(start_episode + 1, cfg.NUM_EPISODES + 1):
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
        
        # Store episode reward and beta
        agent.store_episode_reward(episode_reward)
        episode_rewards.append(episode_reward)
        beta_history.append(agent.beta)
        
        # Log episode information
        agent.adjust_beta()
        print(f"Episode {episode}: Reward = {episode_reward:.2f}, Steps = {episode_steps}")
        
        # Periodic saving and plotting
        if episode % cfg.SAVE_FREQUENCY == 0:
            # Save checkpoint with all necessary info
            checkpoint_path = os.path.join("checkpoints", f"{run_name}_ep{episode}.pt")
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'actor_optimizer': agent.actor_optimizer.state_dict(),
                'critic_optimizer': agent.critic_optimizer.state_dict(),
                'episode': episode,
                'beta': agent.beta,
                'T': agent.T
            }, checkpoint_path)
            
            # Save rewards data
            rewards_data = {
                'algorithm': 'SFAC',
                'run_name': run_name,
                'rewards': episode_rewards,
                'timestamp': datetime.now().strftime('%Y%m%d-%H%M%S')
            }
            rewards_file_path = checkpoint_path.replace(".pt", "_rewards.json")
            with open(rewards_file_path, 'w') as f:
                json.dump(rewards_data, f, indent=4)
                
            # Save beta history
            beta_data = {
                'algorithm': 'SFAC',
                'run_name': run_name,
                'beta_history': beta_history,
                'timestamp': datetime.now().strftime('%Y%m%d-%H%M%S')
            }
            beta_file_path = checkpoint_path.replace(".pt", "_beta.json")
            with open(beta_file_path, 'w') as f:
                json.dump(beta_data, f, indent=4)
            
            # Create and save plots
            plot_path = os.path.join("plots", f"{run_name}_ep{episode}.png")
            plot_learning_curve(episode_rewards, "SFAC", save_path=plot_path)
            
            beta_plot_path = os.path.join("plots", f"{run_name}_beta_ep{episode}.png")
            plot_beta_history(beta_history, save_path=beta_plot_path)
            
            print(f"Checkpoint saved to {checkpoint_path}")
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    # Save final model
    final_checkpoint_path = os.path.join("checkpoints", f"{run_name}_final.pt")
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
        'actor_optimizer': agent.actor_optimizer.state_dict(),
        'critic_optimizer': agent.critic_optimizer.state_dict(),
        'episode': episode,
        'beta': agent.beta,
        'T': agent.T
    }, final_checkpoint_path)
    
    # Save final rewards data
    final_rewards_data = {
        'algorithm': 'SFAC',
        'run_name': run_name,
        'rewards': episode_rewards,
        'timestamp': datetime.now().strftime('%Y%m%d-%H%M%S')
    }
    final_rewards_path = os.path.join("results", f"{run_name}_final_rewards.json")
    with open(final_rewards_path, 'w') as f:
        json.dump(final_rewards_data, f, indent=4)
    
    # Save final beta history
    final_beta_data = {
        'algorithm': 'SFAC',
        'run_name': run_name,
        'beta_history': beta_history,
        'timestamp': datetime.now().strftime('%Y%m%d-%H%M%S')
    }
    final_beta_path = os.path.join("results", f"{run_name}_final_beta.json")
    with open(final_beta_path, 'w') as f:
        json.dump(final_beta_data, f, indent=4)
    
    # Save final plots
    final_plot_path = os.path.join("plots", f"{run_name}_final.png")
    plot_learning_curve(episode_rewards, "SFAC", save_path=final_plot_path)
    
    final_beta_plot_path = os.path.join("plots", f"{run_name}_beta_final.png")
    plot_beta_history(beta_history, save_path=final_beta_plot_path)
    
    env.close()
    eval_env.close()
    
    return agent, episode_rewards, run_name

def train_policy_gradient(checkpoint_path=None):
    """Train a policy gradient agent with checkpoint resumption"""
    env_name_safe = cfg.ENV_NAME.replace('/', '_')
    
    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
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
    
    # Initialize agent
    agent = PolicyGradientAgent(state_dim=state_dim, action_size=action_size, 
                               actor=actor, critic=critic)
    
    # Initialize training variables
    start_episode = 0
    episode_rewards = []
    run_name = None
    
    # Load from checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # Load model weights
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.critic.load_state_dict(checkpoint['critic'])
        
        # Load optimizer states if available
        if 'actor_optimizer' in checkpoint:
            agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        if 'critic_optimizer' in checkpoint:
            agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            
        # Resume episode count
        start_episode = checkpoint['episode']
        print(f"Resuming from episode {start_episode + 1}")
        
        # Extract run name from checkpoint path if possible
        run_name_parts = os.path.basename(checkpoint_path).split('_ep')
        if len(run_name_parts) > 0:
            run_name = run_name_parts[0]
            print(f"Continuing run: {run_name}")
            
        # Try to load reward history
        rewards_file = checkpoint_path.replace(".pt", "_rewards.json")
        if os.path.exists(rewards_file):
            with open(rewards_file, 'r') as f:
                data = json.load(f)
                episode_rewards = data.get('rewards', [])
                print(f"Loaded {len(episode_rewards)} previous episode rewards")
    
    # Create a new run name if not resuming
    if run_name is None:
        run_name = f"PG_{env_name_safe}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        print(f"Starting new run: {run_name}")
    
    # Training variables
    total_steps = 0
    
    # Main training loop
    for episode in range(start_episode + 1, cfg.NUM_EPISODES + 1):
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
                
            # If done, start a new episode
            if done or truncated:
                break
        
        # Store episode reward
        agent.store_episode_reward(episode_reward)
        episode_rewards.append(episode_reward)
        
        # Log episode information
        print(f"Episode {episode}: Reward = {episode_reward:.2f}, Steps = {episode_steps}")
        
        # Periodic saving and plotting
        if episode % cfg.SAVE_FREQUENCY == 0:
            # Save checkpoint with all necessary info
            checkpoint_path = os.path.join("checkpoints", f"{run_name}_ep{episode}.pt")
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'actor_optimizer': agent.actor_optimizer.state_dict(),
                'critic_optimizer': agent.critic_optimizer.state_dict(),
                'episode': episode
            }, checkpoint_path)
            
            # Save rewards data
            rewards_data = {
                'algorithm': 'PG',
                'run_name': run_name,
                'rewards': episode_rewards,
                'timestamp': datetime.now().strftime('%Y%m%d-%H%M%S')
            }
            rewards_file_path = checkpoint_path.replace(".pt", "_rewards.json")
            with open(rewards_file_path, 'w') as f:
                json.dump(rewards_data, f, indent=4)
            
            # Create and save plot
            plot_path = os.path.join("plots", f"{run_name}_ep{episode}.png")
            plot_learning_curve(episode_rewards, "Policy Gradient", save_path=plot_path)
            
            print(f"Checkpoint saved to {checkpoint_path}")
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    # Save final model
    final_checkpoint_path = os.path.join("checkpoints", f"{run_name}_final.pt")
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
        'actor_optimizer': agent.actor_optimizer.state_dict(), 
        'critic_optimizer': agent.critic_optimizer.state_dict(),
        'episode': episode
    }, final_checkpoint_path)
    
    # Save final rewards data
    final_rewards_data = {
        'algorithm': 'PG',
        'run_name': run_name,
        'rewards': episode_rewards,
        'timestamp': datetime.now().strftime('%Y%m%d-%H%M%S')
    }
    final_rewards_path = os.path.join("results", f"{run_name}_final_rewards.json")
    with open(final_rewards_path, 'w') as f:
        json.dump(final_rewards_data, f, indent=4)
    
    # Save final plot
    final_plot_path = os.path.join("plots", f"{run_name}_final.png")
    plot_learning_curve(episode_rewards, "Policy Gradient", save_path=final_plot_path)
    
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train RL agents with checkpoint resumption")
    parser.add_argument("--algorithm", type=str, default=cfg.ALGORITHM, 
                        choices=["sfac", "pg", "both"], 
                        help="Algorithm to train: 'sfac', 'pg', or 'both'")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file for resuming training")
    parser.add_argument("--pg-checkpoint", type=str, default=None,
                        help="Path to Policy Gradient checkpoint (when using 'both')")
    args = parser.parse_args()
    
    # Train SFAC agent
    if args.algorithm == "sfac" or args.algorithm == "both":
        sfac_agent, sfac_rewards, sfac_run_name = train_sfac(checkpoint_path=args.checkpoint)
    
    # Train Policy Gradient agent
    if args.algorithm == "pg" or args.algorithm == "both":
        pg_checkpoint = args.checkpoint if args.algorithm == "pg" else args.pg_checkpoint
        pg_agent, pg_rewards, pg_run_name = train_policy_gradient(checkpoint_path=pg_checkpoint)
    
    # Create comparison plot if both algorithms were trained
    if args.algorithm == "both":
        comparison_path = os.path.join("plots", f"comparison_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png")
        plot_comparison(sfac_rewards=sfac_rewards, pg_rewards=pg_rewards, save_path=comparison_path)
        print(f"Comparison plot saved to {comparison_path}")
    
    print("Training completed!")

if __name__ == "__main__":
    main()