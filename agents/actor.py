import torch
import torch.nn as nn
import numpy as np
from utils import config as cfg
import gymnasium as gym
GAMMA = cfg.GAMMA
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from convol.convolution import Convolution
import random
import ale_py

gym.register_envs(ale_py)

class Actor(nn.Module):
    def __init__(self, state_dim, action_size, hidden1_dim=128, hidden2_dim=128):
        super(Actor, self).__init__()
        
        self.policy_convolution = Convolution(state_dim)
        conv_out_dims = self.policy_convolution.calculate_conv_out_dims(state_dim)
        self.fc1 = nn.Linear(conv_out_dims, hidden1_dim)
        self.policy_mean = nn.Linear(hidden1_dim, action_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, state):
        # Assume state is (batch, H, W, C), convert to (batch, C, H, W)
        if state.dim() == 4 and state.shape[1] == 210:  # likely (N, 210, 160, 3)
            state = state.permute(0, 3, 1, 2)
    
        conv_state = self.policy_convolution.convolute(state)
        x = self.relu(self.fc1(conv_state))
        logits = self.policy_mean(x)
        probs = F.softmax(logits, dim=-1)
        return probs
    
    def evaluate(self, states, actions):
        """
        Evaluate actions given states and return log probabilities, entropy and distribution
        Used for PPO updates
        """
        action_probs = self(states)
        dist = Categorical(action_probs)
        
        # Get log probabilities of actions
        action_log_probs = dist.log_prob(actions.squeeze())
        
        # Calculate entropy for exploration bonus
        entropy = dist.entropy()
        
        return action_log_probs, entropy, dist
        
    def sample_action(self, action_probs):
        """Sample an action from the policy distribution"""
        epsilon = 0.1  # Exploration rate

        if random.random() < epsilon:
            # Uniformly random action from 0 to 6
            action = torch.randint(0, 7, (1,)).item()
            return action

        # Ensure action_probs is valid probability distribution
        if isinstance(action_probs, torch.Tensor):
            # Normalize to ensure probabilities sum to 1
            probs = F.softmax(action_probs, dim=-1) if action_probs.sum() != 1.0 else action_probs
        else:
            # Convert numpy array or list to tensor if needed
            probs = torch.tensor(action_probs)
            probs = F.softmax(probs, dim=-1) if probs.sum() != 1.0 else probs

        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()

        return action

    def flatten_parameters(self):
        params = []
        for param in self.parameters():
            params.append(param)
        return params

    def smoothened_gradient_update(self, states, actions, critic, beta, num_perturbations=5, advantages=None):
        """
        Enhanced smoothened gradient update with dynamic tuning support for on-policy learning
        
        Args:
            states: Batch of states
            actions: Batch of actions
            critic: Critic network
            beta: Smoothing factor (controls the balance between exploration and exploitation)
            num_perturbations: Number of perturbations to use for gradient estimation
            advantages: Advantages for policy gradient (used in on-policy methods)
            
        Returns:
            Smoothened gradient estimates
        """
        # Initialize smoothened gradient accumulator
        smoothened_gradient = [torch.zeros_like(param) for param in self.flatten_parameters()]
        
        # Perturbation scale (smaller c gives finer estimation but may be numerically unstable)
        c = 0.01
        
        # Get initial action probabilities
        action_probs = self(states)
        
        # Compute log probabilities of taken actions
        action_indices = actions
        log_probs = torch.log(torch.gather(action_probs, dim=1, index=action_indices))
        
        # Get value estimates or use provided advantages
        if advantages is None:
            # If advantages not provided, estimate them using critic
            values = critic(states, action_probs).detach()
            # Use values directly as a simple advantage estimate
            advantages = values
        
        # Compute policy loss (negative for gradient ascent)
        # Using PPO-style objective
        original_loss = -torch.mean(advantages * log_probs)
        original_loss.backward(retain_graph=True)

        # Store original gradients
        original_gradients = [param.grad.clone() for param in self.flatten_parameters()]
        
        # Zero out gradients to prepare for gradient estimation
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()
        
        # Estimate second-order effects with perturbations
        for i in range(num_perturbations):
            # Generate perturbation directions (Rademacher distribution: ±1 with equal probability)
            delta_k = [2 * torch.bernoulli(torch.full(param.shape, 0.5)).float() - 1 for param in self.flatten_parameters()]
            
            # Create perturbed parameter sets (positive and negative perturbations)
            theta1 = [param + c * delta_k_i for param, delta_k_i in zip(self.flatten_parameters(), delta_k)]
            theta2 = [param - c * delta_k_i for param, delta_k_i in zip(self.flatten_parameters(), delta_k)]

            # Ensure parameters require gradients
            theta1 = [param.clone().detach().requires_grad_(True) for param in theta1]
            theta2 = [param.clone().detach().requires_grad_(True) for param in theta2]

            # Compute gradients at perturbed points (using on-policy objective)
            grad_theta1 = self.compute_on_policy_gradients(states, actions, advantages, critic, theta1)
            grad_theta2 = self.compute_on_policy_gradients(states, actions, advantages, critic, theta2)

            # Calculate the directional derivative
            delta_G_k = [g1 - g2 for g1, g2 in zip(grad_theta1, grad_theta2)]
            
            # Generate random direction for Gaussian smoothing
            rho_k = [torch.randn_like(param) for param in self.flatten_parameters()]
            
            # For numerical stability, normalize rho_k
            rho_norm = torch.sqrt(sum(torch.sum(r**2) for r in rho_k))
            rho_k = [r / rho_norm for r in rho_k]
            
            # Compute dot products for the smoothing
            grad_dot_rho = sum(torch.sum(g * r) for g, r in zip(original_gradients, rho_k))
            
            # Compute second-order term components
            second_order_terms = []
            for j in range(len(delta_k)):
                delta_k_dot_rho = torch.sum(delta_k[j] * rho_k[j])
                delta_g_dot_rho = torch.sum(delta_G_k[j] * rho_k[j])
                
                second_order_term = 0.5 * beta ** 2 * (delta_k_dot_rho * delta_g_dot_rho)
                second_order_terms.append(second_order_term)
            
            # Update the smoothened gradient with this perturbation's contribution
            for j, r in enumerate(rho_k):
                smoothened_gradient[j] += r * (original_loss + beta * grad_dot_rho + second_order_terms[j])
        
        # Scale the smoothened gradient by the number of perturbations and beta
        smoothened_gradient = [sg / (num_perturbations * beta) for sg in smoothened_gradient]
        
        return smoothened_gradient

    def compute_on_policy_gradients(self, states, actions, advantages, critic, theta=None):
        """
        Compute policy gradients for on-policy learning
        
        Args:
            states: Batch of states
            actions: Batch of actions
            advantages: Advantage estimates
            critic: Critic network
            theta: Optional parameter set to use (for perturbation-based estimation)
            
        Returns:
            Computed gradients
        """
        # Save original parameters if theta is provided
        if theta is not None:
            original_params = [param.clone() for param in self.flatten_parameters()]
            original_grads = [param.grad.clone() if param.grad is not None else None for param in self.flatten_parameters()]

            # Temporarily set parameters to theta
            for param, theta_param in zip(self.flatten_parameters(), theta):
                param.data.copy_(theta_param)
        
        # Clear gradients
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()
                
        # Forward pass with current parameters
        action_probs = self(states)
        action_indices = actions
        log_probs = torch.log(torch.gather(action_probs, dim=1, index=action_indices))
        
        # Compute policy loss (negative for gradient ascent)
        policy_loss = -torch.mean(advantages * log_probs)
        
        # Add entropy bonus for exploration
        entropy = -torch.mean(torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=1))
        entropy_bonus = 0.01 * entropy  # Small coefficient for entropy
        
        # Total loss with entropy bonus
        loss = policy_loss - entropy_bonus
        
        # Compute gradients
        loss.backward()
        
        # Get the gradients
        gradients = [param.grad.clone() for param in self.parameters()]
        
        # Restore original parameters and gradients if needed
        if theta is not None:
            for param, original_param, original_grad in zip(self.flatten_parameters(), original_params, original_grads):
                param.data.copy_(original_param)
                if original_grad is not None:
                    param.grad = original_grad
        
        return gradients

    def compute_rollout(self, env=None, max_steps=1000):
        """
        Run a single rollout for the current policy
        
        Args:
            env: Environment to use (creates a new one if None)
            max_steps: Maximum number of steps per episode
            
        Returns:
            states, actions, rewards, log_probs, dones, episode_reward
        """
        if env is None:
            env_temp = gym.make('ALE/Assault-v5', render_mode='rgb_array')
        else:
            env_temp = env
            
        states = []
        actions = []
        rewards = []
        log_probs = []
        dones = []
        
        state, _ = env_temp.reset()
        done = False
        episode_reward = 0
        
        for step in range(max_steps):
            states.append(state)
            
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Get action probabilities
            action_probs = self(state_tensor)
            
            # Sample action
            action = self.sample_action(action_probs[0])
            actions.append(action)
            
            # Get log probability
            action_idx = torch.tensor([[action]]).to(device)
            log_prob = torch.log(torch.gather(action_probs, 1, action_idx)).item()
            log_probs.append(log_prob)
            
            # Take action in environment
            next_state, reward, done, truncated, _ = env_temp.step(action)
            
            # Store transition
            rewards.append(reward)
            dones.append(done or truncated)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            if done or truncated:
                break
                
        # Close environment if we created it
        if env is None:
            env_temp.close()
            
        return states, actions, rewards, log_probs, dones, episode_reward