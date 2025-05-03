from .actor import Actor
from .critic import Critic
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import config as cfg
from collections import deque
import math

# Configuration parameters
GAMMA = cfg.GAMMA
TAU = cfg.TAU
LR_ACTOR = cfg.LR_ACTOR
LR_CRITIC = cfg.LR_CRITIC
action_size = cfg.action_size

# SFAC-specific parameters
T_MIN = 5        # Minimum number of perturbations
T_MAX = 1000     # Maximum number of perturbations
T_INIT = 100     # Initial number of perturbations
BETA_MIN = 0.01  # Minimum smoothing parameter
BETA_MAX = 5.0   # Maximum smoothing parameter
BETA_INIT = 1.0  # Initial smoothing parameter
ETA = 0.1        # Learning rate for T adjustment
LAMBDA = 0.5     # Learning rate for beta adjustment
KAPPA = 0.005    # Target critic update rate
TARGET_VARIANCE = 0.1  # Target gradient variance
REWARD_BUFFER_SIZE = 20  # Size of reward buffer for reward improvement calculation
NUM_TRAJECTORIES = 5     # Number of trajectories to collect before each update


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
    
    def calculate_analytical_gradient_variance(self, gradient):
        """
        Calculate gradient variance analytically using Taylor expansion
        
        σ_g^2 ≈ β^2 · Tr(∇J(θ)∇J(θ)^T) = β^2 · ||∇J(θ)||_2^2
        """
        # Flatten and concatenate all gradients
        flat_gradient = torch.cat([g.view(-1) for g in gradient])
        
        # Calculate L2 norm squared
        gradient_norm_squared = torch.sum(flat_gradient ** 2).item()
        
        # Analytical variance
        variance = self.beta ** 2 * gradient_norm_squared
        
        return variance
    
    def adjust_T(self, gradient_variance):
        """
        Adjust number of perturbations based on gradient variance
        
        T ← clip(T · exp(η · (σ_g^2 - τ_target)), T_min, T_max)
        """
        # Adjust T exponentially based on difference from target variance
        adjustment = math.exp(ETA * (gradient_variance - TARGET_VARIANCE))
        new_T = int(self.T * adjustment)
        
        # Clip T to valid range
        self.T = max(T_MIN, min(T_MAX, new_T))
        
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
        with torch.no_grad():
            # Get next actions from current policy (on-policy)
            next_actions_probs = self.actor(next_states)
            # Get target Q values using target critic
            target_q_values = rewards + (1 - dones) * GAMMA * self.target_critic(next_states, next_actions_probs)

        # Get current Q values
        current_q_values = self.critic(states, actions_one_hot)
        
        # Compute critic loss
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----- Update Actor using Smoothed Gradient with Taylor Expansion -----
        # 1. Get baseline performance and gradient
        self.actor_optimizer.zero_grad()
        action_probs = self.actor(states)
        
        # Original value (negative for gradient ascent)
        J_theta = -torch.mean(self.critic(states, action_probs))
        J_theta.backward(retain_graph=True)
        
        # Store original gradient
        original_gradient = [param.grad.clone() for param in self.actor.parameters()]
        
        # Calculate analytical gradient variance
        gradient_variance = self.calculate_analytical_gradient_variance(original_gradient)
        
        # Adjust number of perturbations
        self.adjust_T(gradient_variance)
        
        # 2. Sample perturbations and calculate smoothed gradient
        smoothed_gradient = [torch.zeros_like(param) for param in self.actor.parameters()]
        
        for i in range(self.T):
            # Sample Gaussian perturbation
            rho_i = [torch.randn_like(param) for param in self.actor.parameters()]
            
            # Normalize the perturbation
            rho_norm = torch.sqrt(sum(torch.sum(r**2) for r in rho_i))
            rho_i = [r / rho_norm for r in rho_i]
            
            # Perturb parameters
            perturbed_params = []
            for param, rho in zip(self.actor.parameters(), rho_i):
                perturbed_params.append(param + self.beta * rho)
            
            # Compute gradient at perturbed point
            self.actor_optimizer.zero_grad()
            
            # Save original parameters
            original_params = [p.clone() for p in self.actor.parameters()]
            
            # Set perturbed parameters
            for param, perturbed_param in zip(self.actor.parameters(), perturbed_params):
                param.data.copy_(perturbed_param)
            
            # Get performance at perturbed point
            perturbed_action_probs = self.actor(states)
            J_theta_i = -torch.mean(self.critic(states, perturbed_action_probs))
            J_theta_i.backward(retain_graph=True)
            
            # Get gradient at perturbed point
            perturbed_gradient = [param.grad.clone() for param in self.actor.parameters()]
            
            # Calculate gradient difference for Hessian approximation
            delta_g_i = [pg - og for pg, og in zip(perturbed_gradient, original_gradient)]
            
            # Restore original parameters
            for param, orig_param in zip(self.actor.parameters(), original_params):
                param.data.copy_(orig_param)
            
            # Calculate linear term: ρ_i^T ∇J(θ)
            linear_term = sum(torch.sum(r * g) for r, g in zip(rho_i, original_gradient))
            
            # Calculate second-order term: ρ_i^T H ρ_i ≈ ρ_i^T (δg_i) / β
            second_order_term = sum(torch.sum(r * dg) / self.beta for r, dg in zip(rho_i, delta_g_i))
            
            # Taylor expansion approximation
            taylor_approx = J_theta + self.beta * linear_term + 0.5 * self.beta**2 * second_order_term
            
            # Contribute to smoothed gradient
            for j, (r, dg) in enumerate(zip(rho_i, delta_g_i)):
                # Combine contribution from zeroth, first, and second order terms
                smoothed_gradient[j] += r * (taylor_approx + self.beta * linear_term + 
                                            0.5 * self.beta**2 * torch.sum(r * dg))
            
        # Scale the smoothed gradient
        smoothed_gradient = [sg / (self.T * self.beta) for sg in smoothed_gradient]
        
        # Apply smoothed gradient
        self.actor_optimizer.zero_grad()
        for param, smoothed_grad in zip(self.actor.parameters(), smoothed_gradient):
            param.grad = smoothed_grad
        
        # Update actor
        self.actor_optimizer.step()
        
        # Update target networks
        self.update_target_networks()
        
        # Calculate average reward from trajectories
        avg_trajectory_reward = sum(self.trajectory_rewards) / len(self.trajectory_rewards)
        
        # Adjust beta based on reward progression
        self.store_episode_reward(avg_trajectory_reward)
        self.adjust_beta()
        
        # Clear trajectory buffer after update
        self.reset_buffers()
        
        return critic_loss.item()