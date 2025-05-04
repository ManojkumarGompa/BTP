from .actor import Actor
from .critic import Critic
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import math
from utils import config as cfg


# Configuration parameters
GAMMA = cfg.GAMMA
TAU = cfg.TAU
LR_ACTOR = cfg.LR_ACTOR
LR_CRITIC = cfg.LR_CRITIC
action_size = cfg.action_size

# SFAC-specific parameters
T_MIN = cfg.T_MIN        # Minimum number of perturbations
T_MAX = cfg.T_MAX        # Maximum number of perturbations
T_INIT = cfg.T_INIT      # Initial number of perturbations
BETA_MIN = cfg.BETA_MIN  # Minimum smoothing parameter
BETA_MAX = cfg.BETA_MAX  # Maximum smoothing parameter
BETA_INIT = cfg.BETA_INIT  # Initial smoothing parameter
ETA = cfg.ETA            # Learning rate for T adjustment
LAMBDA = cfg.LAMBDA      # Learning rate for beta adjustment
KAPPA = cfg.KAPPA        # Target critic update rate
TARGET_VARIANCE = cfg.TARGET_VARIANCE  # Target gradient variance
REWARD_BUFFER_SIZE = cfg.REWARD_BUFFER_SIZE  # Size of reward buffer for reward improvement calculation
NUM_TRAJECTORIES = cfg.NUM_TRAJECTORIES     # Number of trajectories to collect before each update

delta_T_up = cfg.DELTA_T_UP  # Increment step for T
delta_T_down = cfg.DELTA_T_DOWN  # Decrement step for T


class MultiTrajectorySFACAgent:
    def __init__(self, state_dim, action_size):
        self.actor = Actor(state_dim, action_size).float()
        self.critic = Critic(state_dim, action_size).float()
        self.target_actor = Actor(state_dim, action_size).float()
        self.target_critic = Critic(state_dim, action_size).float()
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        # On-policy trajectory buffers
        self.trajectory_count = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
        # Initialize target networks with current parameters
        self.update_target_networks(tau=1.0)
        
        # SFAC hyperparameters
        self.T = T_INIT  # Number of perturbations
        self.beta = BETA_INIT  # Smoothing parameter
        
        # Reward buffer for dynamic adjustments
        self.reward_buffer = deque(maxlen=REWARD_BUFFER_SIZE)
        
        # Trajectory rewards
        self.trajectory_rewards = []
        
    def update_target_networks(self, tau=TAU):
        """Soft update target networks"""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(KAPPA * param.data + (1.0 - KAPPA) * target_param.data)
            
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
        """Store episode reward in the reward buffer for beta adjustment"""
        self.reward_buffer.append(episode_reward)
    def calculate_analytical_gradient_variance(self, J_theta, gradient, sfac_gradient):
        """
        Calculate gradient variance analytically using the corrected formula:
        
        σ_g^2 = (1 / (T - 1)) * Σ_{i=1}^T ∥βρ_i (J(θ) + (βρ_i / T) ∇θJ(θ)) - ∇θSFACJ(θ)∥^2
        """
        T = self.T
        beta = self.beta
        # Print the shapes of sfac_gradient and gradient


        # Initialize variance accumulator
        variance_accumulator = 0.0

        for i in range(T):
            # Sample perturbation ρ_i (Gaussian noise)
            print(f"i: {i}, T: {T}")
            rho = [torch.randn_like(param) for param in self.actor.parameters()]

            # Compute the variance term for this perturbation
            gradient_difference = [
                beta * r * (J_theta + (beta / T) * g.item()) - sfac_g.item()
                for r, g, sfac_g in zip(rho, gradient, sfac_gradient)
            ]
            gradient_difference_norm = sum(
                torch.sum(diff ** 2) for diff in gradient_difference
            )
            variance_accumulator += gradient_difference_norm

        # Final variance calculation
        gradient_variance = variance_accumulator / (T - 1)
        return gradient_variance
    # def calculate_analytical_gradient_variance(self, gradient):
    #     """
    #     Calculate gradient variance analytically using Taylor expansion
        
    #     σ_g^2 ≈ β^2 · Tr(∇J(θ)∇J(θ)^T) = β^2 · ||∇J(θ)||_2^2
    #     """
    #     # Flatten and concatenate all gradients
    #     flat_gradient = torch.cat([g.view(-1) for g in gradient])
        
    #     # Calculate L2 norm squared
    #     gradient_norm_squared = torch.sum(flat_gradient ** 2).item()
        
    #     # Analytical variance
    #     variance = self.beta ** 2 * gradient_norm_squared
        
    #     return variance
    
    # def adjust_T(self, gradient_variance):
    #     """
    #     Adjust number of perturbations based on gradient variance
        
    #     T ← clip(T · exp(η · (σ_g^2 - τ_target)), T_min, T_max)
    #     """
    #     # Adjust T exponentially based on difference from target variance
    #     adjustment = math.exp(ETA * (gradient_variance - TARGET_VARIANCE))
    #     new_T = int(self.T * adjustment)
        
    #     # Clip T to valid range
    #     self.T = max(T_MIN, min(T_MAX, new_T))
        
    #     print(f"Adjusted T to {self.T} based on variance: {gradient_variance:.4f}")
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
            self.T = min(self.T + delta_T_up, T_MAX)
        elif gradient_variance < TARGET_VARIANCE:
            # Too stable, decrease T
            self.T = max(self.T - delta_T_down, T_MIN)
        
        print(f"Adjusted T to {self.T} based on variance: {gradient_variance:.4f}")
    
    def adjust_beta(self):
        """
        Adjust beta based on reward improvement
        
        β ← clip(β · exp(-λ · sign(ΔR)), β_min, β_max)
        """
        if len(self.reward_buffer) < REWARD_BUFFER_SIZE // 2:
            return  # Not enough data
        
        # Calculate reward improvement
        k = REWARD_BUFFER_SIZE // 2
        recent_rewards = list(self.reward_buffer)[-k:]
        previous_rewards = list(self.reward_buffer)[-(2*k):-k]
        
        if not previous_rewards:
            return  # Not enough historical data
        
        # Calculate average rewards
        recent_avg = sum(recent_rewards) / len(recent_rewards)
        previous_avg = sum(previous_rewards) / len(previous_rewards)
        
        # Reward improvement
        delta_R = recent_avg - previous_avg
        
        # Adjust beta based on sign of reward improvement
        # If rewards are improving (delta_R > 0), decrease beta (more exploitation)
        # If rewards are declining (delta_R < 0), increase beta (more exploration)
        adjustment = math.exp(-LAMBDA * np.sign(delta_R))
        new_beta = self.beta * adjustment
        
        # Clip beta to valid range
        self.beta = max(BETA_MIN, min(BETA_MAX, new_beta))
        
        print(f"Adjusted beta to {self.beta:.4f} based on reward change: {delta_R:.2f}")
    
    def has_enough_trajectories(self):
        """Check if enough trajectories have been collected for an update"""
        return self.trajectory_count >= NUM_TRAJECTORIES
    
    def update(self):
        """Update policy and value functions using multiple trajectories with SFAC"""
        # If no transitions stored, skip update
        if len(self.states) == 0:
            return 0
        
        # Convert stored trajectories to tensors
        states = torch.FloatTensor(np.stack(self.states))
        actions = torch.LongTensor(self.actions).unsqueeze(1)
        rewards = torch.FloatTensor(self.rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.stack(self.next_states))
        dones = torch.FloatTensor(self.dones).unsqueeze(1)

        # Create one-hot actions for critic input
        actions_one_hot = torch.zeros(len(self.actions), action_size).scatter(1, actions, 1)

        # ----- Update Critic -----
        print("updating the critic")
        with torch.no_grad():
            # Get next actions from current policy (on-policy)
            target_next_actions = self.target_actor(next_states)
            # Get target Q values using target critic
            target_q_values = rewards + (1 - dones) * GAMMA * self.target_critic(next_states, target_next_actions)

        # Get current Q values
        current_q_values = self.critic(states, actions_one_hot)
        
        # Compute critic loss
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

   

        
        # Calculate analytical gradient variance
        print("updating the actor")
        self.actor_optimizer.zero_grad()
        smoothened_gradient,original_gradient,original_loss=self.actor.smoothened_gradient_update(states,actions,self.critic,self.beta,self.T)
        # gradient_variance = self.calculate_analytical_gradient_variance(-1*original_loss, original_gradient,smoothened_gradient)
        # print(f"Gradient variance: {gradient_variance:.4f}")
        
        # Adjust number of perturbations
        # self.adjust_T(gradient_variance)
        
     
        self.assign_flat_list_to_param_grads(smoothened_gradient)
        
        # Update actor
        self.actor_optimizer.step()
        
        # Update target networks
        self.update_target_networks()
        
        # Calculate average reward from trajectories
        avg_trajectory_reward = sum(self.trajectory_rewards) / len(self.trajectory_rewards)
        
        # Adjust beta based on reward progression
        self.store_episode_reward(avg_trajectory_reward)
        self.adjust_beta()
        
      
        print("updated the agent(actor and critic)")
        
        return critic_loss.item()
    def assign_flat_list_to_param_grads(self, flat_grad_list):
        """
        Assign a flat list of gradient tensors (scalars or small tensors)
        to the .grad fields of model parameters by reshaping appropriately.
        """
        idx = 0
        for param in self.actor.parameters():
            numel = param.numel()
            # Collect next `numel` gradient scalars
            flat_slice = flat_grad_list[idx:idx + numel]
            # Stack and reshape to match the parameter's shape
            reshaped = torch.stack(flat_slice).view_as(param)
            param.grad = reshaped.clone()
            idx += numel

