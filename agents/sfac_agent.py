import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from collections import deque
from utils import config as cfg
import random
from memory_profiler import profile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants for T adjustment
T_MIN = 20
T_MAX = 500
TARGET_VARIANCE = 0.01
DELTA_T_UP = 100
DELTA_T_DOWN = 50

class MultiTrajectorySFACAgent:
    def __init__(self, state_dim, action_size, actor=None, critic=None):
        from agents.actor import Actor
        from agents.critic import Critic
        
        # Initialize actor and critic if not provided
        self.actor = actor if actor is not None else Actor(state_dim, action_size).to(device)
        self.critic = critic if critic is not None else Critic(state_dim, action_size).to(device)
        
        # Initialize optimizers with AdamW as recommended
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
        
        # Initialize parameters from config
        self.beta = cfg.BETA_INIT  # Smoothing parameter
        self.T = cfg.T_INIT  # Number of perturbations
        
        # Initialize buffers for transitions and rewards
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
        # Initialize trajectory tracking
        self.trajectory_count = 0
        self.trajectory_rewards = []
        
        # Initialize reward buffer for beta adjustment
        self.reward_buffer = deque(maxlen=cfg.REWARD_BUFFER_SIZE)
        
    def reset_buffers(self):
        """Reset all buffers"""
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
        """Store episode reward in the reward buffer for beta adjustment"""
        self.reward_buffer.append(episode_reward)
        
    def calculate_advantages(self, rewards, values, next_values, dones, gamma=cfg.GAMMA, lam=cfg.GAE_LAMBDA):
        """
        Calculate Generalized Advantage Estimation (GAE) with lambda parameter.
        
        Args:
            rewards: List of rewards
            values: List of value estimates for states
            next_values: List of value estimates for next states
            dones: List of done flags
            gamma: Discount factor
            lam: GAE lambda parameter
            
        Returns:
            advantages: Tensor of advantage estimates
            returns: Tensor of target returns for value function
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
            
            # Clip advantages as recommended
            std = advantages.std()
            advantages = torch.clamp(advantages, -5 * std, 5 * std)
        
        return advantages, returns
    
    def calculate_analytical_gradient_variance(self, J_theta, gradient, sfac_gradient):
        """
        Calculate gradient variance analytically using the corrected formula:
        
        σ_g^2 = (1 / (T - 1)) * Σ_{i=1}^T ∥βρ_i (J(θ) + (βρ_i / T) ∇θJ(θ)) - ∇θSFACJ(θ)∥^2
        """
        T = self.T
        beta = self.beta

        # Initialize variance accumulator
        variance_accumulator = 0.0

        for i in range(T):
            # Sample perturbation ρ_i (Gaussian noise)
            rho = torch.randn_like(gradient)

            # Compute the variance term for this perturbation
            if isinstance(sfac_gradient, list):
                # Convert list of tensors to a single tensor
                sfac_gradient_tensor = torch.cat([sg.view(-1) for sg in sfac_gradient])
            else:
                sfac_gradient_tensor = sfac_gradient

            gradient_difference = beta * rho * (J_theta + (beta / T) * gradient) - sfac_gradient_tensor
            gradient_difference_norm_squared = torch.sum(gradient_difference ** 2).item()
            variance_accumulator += gradient_difference_norm_squared

        # Final variance calculation
        gradient_variance = variance_accumulator / (T - 1) if T > 1 else variance_accumulator
        return gradient_variance
    
    def adjust_T(self, gradient_variance):
        """
        Adjust number of perturbations based on gradient variance.

        If σ_g^2 > TARGET_VARIANCE (too noisy):
            T ← min(T + ΔT_up, T_max)
        If σ_g^2 < TARGET_VARIANCE (too stable):
            T ← max(T - ΔT_down, T_min)
        """
        if gradient_variance > TARGET_VARIANCE:
            # Too noisy, increase T
            self.T = min(self.T + DELTA_T_UP, T_MAX)
        elif gradient_variance < TARGET_VARIANCE:
            # Too stable, decrease T
            self.T = max(self.T - DELTA_T_DOWN, T_MIN)
        
        print(f"Adjusted T to {self.T} based on variance: {gradient_variance:.4f}")
    
    def adjust_beta(self):
        """
        Adaptive adjustment of beta based on reward trends.
        Increase if improving, decrease if plateauing or declining.
        """
        if len(self.reward_buffer) < 10:
            return  # Not enough data
            
        # Get recent and past rewards
        recent_rewards = list(self.reward_buffer)[-5:]
        past_rewards = list(self.reward_buffer)[-10:-5]
        
        # Calculate averages
        recent_avg = sum(recent_rewards) / len(recent_rewards)
        past_avg = sum(past_rewards) / len(past_rewards)
        
        # Replace with delta-based adjustment
        delta = (recent_avg - past_avg) / (abs(past_avg) + 1e-8)
        if delta > 0.1:  # 10%+ improvement
            self.beta = max(cfg.BETA_MIN, self.beta * 0.95)
        elif delta < -0.1:  # 10%+ decline
            self.beta = min(cfg.BETA_MAX, self.beta * 1.05)
        else:  # Stable phase
            self.beta = self.beta * 0.997  # Slow decay
            # @profile    
    def update(self, batch_size=cfg.BATCH_SIZE, n_epochs=cfg.N_EPOCHS):
        """
        Update actor and critic networks using collected transitions and SFAC.
        Uses multiple epochs like PPO for better data efficiency.
        """
        if len(self.states) < batch_size:
            return  # Not enough data
            
        # Convert lists to tensors
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
        # No need to convert to tensor again since calculate_advantages already returns tensors
        # Just make sure they are on the right device
        if advantages.device != device:
            advantages = advantages.to(device)
        if returns.device != device:
            returns = returns.to(device)
        # advantages = torch.FloatTensor(advantages).to(device)
        # returns = torch.FloatTensor(returns).to(device)
        
        # Multiple epochs of training, like PPO
        for epoch in range(n_epochs):
            # Process in minibatches
            # print(f"Epoch {epoch + 1}/{n_epochs}")
            n_minibatches = max(1, len(self.states) // batch_size)
            indices = np.arange(len(self.states))
            np.random.shuffle(indices)
            
            for batch_idx in range(n_minibatches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(self.states))
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_next_states = next_states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Update critic
                self.critic_optimizer.zero_grad()
                # batch_action_probs = self.actor(batch_states)
                with torch.no_grad():
                    batch_action_probs = self.actor(batch_states).detach()  # No gradient flow
                batch_values = self.critic(batch_states, batch_action_probs)
                critic_loss = nn.MSELoss()(batch_values, batch_returns.unsqueeze(1))
                critic_loss.backward()
                # Apply gradient clipping to critic
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.critic_optimizer.step()
                
                # Update actor using SFAC
                sfac_gradient, original_gradient, original_loss = self.actor.smoothened_gradient_update(
                    batch_states, batch_actions, self.critic,
                    beta=self.beta, num_perturbations=self.T, advantages=batch_advantages
                )
                
                # Apply smoothened gradient
                self.actor_optimizer.zero_grad()
                
                # Set gradients for parameters based on smoothened gradient
                if isinstance(sfac_gradient, torch.Tensor):
                    idx = 0
                    for param in self.actor.parameters():
                        num_params = param.numel()
                        param.grad = sfac_gradient[idx:idx+num_params].view(param.shape)
                        idx += num_params
                else:
                    for param, sg in zip(self.actor.parameters(), sfac_gradient):
                        param.grad = sg.view_as(param)
                
                # Apply gradient clipping to actor
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.actor_optimizer.step()
                
                # # Calculate gradient variance and adjust T periodically
                # if batch_idx == 0 and epoch == 0:  # Only once per update
                #     gradient_variance = self.calculate_analytical_gradient_variance(
                #         original_loss.item(), original_gradient, sfac_gradient
                #     )
                #     self.adjust_T(gradient_variance)
            del batch_states, batch_next_states, batch_actions, batch_advantages
        
        # Reset buffers after multiple epochs of updates
        self.reset_buffers()