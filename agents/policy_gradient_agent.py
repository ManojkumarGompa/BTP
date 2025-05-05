from .actor import Actor
from .critic import Critic
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import config as cfg
from collections import deque
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyGradientAgent:
    def __init__(self, state_dim, action_size, actor=None, critic=None):
        # Initialize actor and critic if not provided
        self.actor = actor if actor is not None else Actor(state_dim, action_size).to(device)
        self.critic = critic if critic is not None else Critic(state_dim, action_size).to(device)
        
        # Initialize optimizers with AdamW as recommended for better performance
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(), 
            lr=cfg.LR_ACTOR,
            weight_decay=1e-5
        )
        self.critic_optimizer = optim.AdamW(
            self.critic.parameters(), 
            lr=cfg.LR_CRITIC,
            weight_decay=1e-5
        )
        
        # On-policy trajectory buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
        # Initialize reward tracking
        self.trajectory_count = 0
        self.trajectory_rewards = []
        self.reward_buffer = deque(maxlen=cfg.REWARD_BUFFER_SIZE)
        
    def reset_buffers(self):
        """Clear trajectory buffers"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.trajectory_count = 0
        self.trajectory_rewards = []
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the on-policy buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def store_trajectory_reward(self, trajectory_reward):
        """Store the reward from a complete trajectory"""
        self.trajectory_rewards.append(trajectory_reward)
        self.trajectory_count += 1
    
    def store_episode_reward(self, episode_reward):
        """Store episode reward in the reward buffer"""
        self.reward_buffer.append(episode_reward)
    
    def calculate_advantages(self, rewards, values, next_values, dones, gamma=cfg.GAMMA, lam=cfg.GAE_LAMBDA):
        """
        Calculate Generalized Advantage Estimation (GAE) with lambda parameter.
        Maintains consistency with SFAC implementation.
        """
        advantages = []
        returns = []
        gae = 0
        
        # Calculate returns and advantages
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t+1]
                
            # TD error
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            
            # Add to lists
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        # Convert to tensors
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Clip advantages for stability
            std = advantages.std()
            advantages = torch.clamp(advantages, -5 * std, 5 * std)
        
        return advantages, returns
            
    def update(self, batch_size=cfg.BATCH_SIZE, n_epochs=cfg.N_EPOCHS):
        """
        Update policy and value functions using regular policy gradient
        with multiple epochs for data efficiency (similar to SFAC)
        """
        # If not enough data, skip update
        if len(self.states) < batch_size:
            return
        
        # Convert stored trajectories to tensors
        states = torch.FloatTensor(np.array(self.states)).to(device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(device)
        actions = torch.LongTensor(np.array(self.actions)).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(np.array(self.rewards)).unsqueeze(1).to(device)
        dones = torch.FloatTensor(np.array(self.dones)).unsqueeze(1).to(device)
        
        # Pre-compute action probabilities and values to save computation
        with torch.no_grad():
            action_probs = self.actor(states)
            next_action_probs = self.actor(next_states)
            values = self.critic(states, action_probs)
            next_values = self.critic(next_states, next_action_probs)
        
        # Calculate advantages and returns once
        advantages, returns = self.calculate_advantages(
            rewards.cpu().detach().numpy().flatten(),
            values.cpu().detach().numpy().flatten(),
            next_values.cpu().detach().numpy().flatten(),
            dones.cpu().detach().numpy().flatten()
        )
        advantages = torch.FloatTensor(advantages).to(device)
        returns = torch.FloatTensor(returns).to(device)
        
        # Multiple epochs of training for better data efficiency
        total_critic_loss = 0
        total_actor_loss = 0
        
        for epoch in range(n_epochs):
            # Process in minibatches
            n_minibatches = max(1, len(self.states) // batch_size)
            indices = np.arange(len(self.states))
            np.random.shuffle(indices)
            
            for batch_idx in range(n_minibatches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(self.states))
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Update critic
                self.critic_optimizer.zero_grad()
                with torch.no_grad():
                    batch_action_probs = self.actor(batch_states).detach() 
                batch_values = self.critic(batch_states, batch_action_probs)
                critic_loss = nn.MSELoss()(batch_values, batch_returns.unsqueeze(1))
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.critic_optimizer.step()
                
                total_critic_loss += critic_loss.item()
                
                # Update actor using policy gradient
                self.actor_optimizer.zero_grad()
                batch_action_probs = self.actor(batch_states)
                log_probs = torch.log(torch.gather(batch_action_probs, 1, batch_actions))
                
                # Add entropy regularization for exploration
                entropy = -torch.mean(torch.sum(batch_action_probs * 
                                               torch.log(batch_action_probs + 1e-10), dim=1))
                
                # Policy loss with entropy bonus
                actor_loss = -torch.mean(log_probs * batch_advantages.unsqueeze(1)) - cfg.ENTROPY_COEF * entropy
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.actor_optimizer.step()
                
                total_actor_loss += actor_loss.item()
            
            # Memory cleanup after each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Final memory cleanup
        del states, next_states, actions, rewards, dones, advantages, returns
        gc.collect()
        
        # Reset buffers after update
        self.reset_buffers()
        
        # Return average losses
        avg_critic_loss = total_critic_loss / (n_epochs * n_minibatches)
        avg_actor_loss = total_actor_loss / (n_epochs * n_minibatches)
        return avg_critic_loss, avg_actor_loss