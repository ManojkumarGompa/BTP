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
from utils.preprocessing import AtariPreprocessor

env = gym.make("ALE/Assault-v5", render_mode='rgb_array')
gym.register_envs(ale_py)
from memory_profiler import profile

class Actor(nn.Module):
    def __init__(self, state_dim, action_size, hidden1_dim=512, hidden2_dim=256):
        super(Actor, self).__init__()
        
        # Enhanced network architecture
        self.policy_convolution = Convolution(state_dim)
        conv_out_dims = self.policy_convolution.calculate_conv_out_dims(state_dim)
        
        # Increased layer sizes as suggested
        self.fc1 = nn.Linear(conv_out_dims, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.policy_mean = nn.Linear(hidden2_dim, action_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights orthogonally for better training
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, state):
        """
        Forward pass through the actor network
        """
        # Ensure state is correctly shaped (batch, channels, height, width)
        if state.dim() == 4 and state.shape[1] == 210:
            state = state.permute(0, 3, 1, 2)
        
        # Pass through convolution layers
        conv_state = self.policy_convolution(state)
        
        # Pass through fully connected layers with dropout
        x = self.relu(self.fc1(conv_state))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        
        # Output action probabilities with clipping to prevent extreme values
        logits = self.policy_mean(x)
        # Clip to prevent extreme values that could cause NaNs in softmax
        logits = torch.clamp(logits, min=-20, max=20)
        probs = F.softmax(logits, dim=-1)
        
        return probs
            
    def evaluate(self, states, actions):
        """
        Evaluate actions given states and return log probabilities, entropy and distribution
        """
        action_probs = self(states)
        dist = Categorical(action_probs)
        
        # Get log probabilities of actions
        action_log_probs = dist.log_prob(actions.squeeze())
        
        # Calculate entropy for exploration bonus
        entropy = dist.entropy()
        
        return action_log_probs, entropy, dist
        
    def sample_action(self, action_probs):
        """
        Sample an action from the policy distribution with epsilon-greedy exploration
        """
        epsilon = 0.1  # Exploration rate
        
        if random.random() < epsilon:
            # Random action based on action space
            action = torch.randint(0, action_probs.size(-1), (1,)).item()
            return action
        
        # Check for NaN values in action probabilities
        if torch.isnan(action_probs).any():
            print("Warning: NaN detected in action probabilities. Using uniform distribution instead.")
            # Use uniform distribution as fallback
            probs = torch.ones_like(action_probs) / action_probs.size(-1)
        else:
            # Ensure action_probs is valid probability distribution
            if isinstance(action_probs, torch.Tensor):
                probs = F.softmax(action_probs, dim=-1) if action_probs.sum() != 1.0 else action_probs
            else:
                probs = torch.tensor(action_probs)
                probs = F.softmax(probs, dim=-1) if probs.sum() != 1.0 else probs
        
        # Add a small epsilon to avoid zeros and rescale
        probs = probs + 1e-10
        probs = probs / probs.sum()
        
        dist = Categorical(probs)
        action = dist.sample().item()
        
        return action
    def flatten_parameters(self):
        """
        Flatten all parameters into a single vector.
        """
        return torch.cat([param.view(-1) for param in self.parameters()])
    
    # @profile    
    def smoothened_gradient_update(self, states, actions, critic, beta, num_perturbations=5, advantages=None):
        """
        Enhanced smoothened gradient update with dynamic tuning support for on-policy learning.
        Preserves the core SFAC algorithm but adds improvements.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            critic: Critic network
            beta: Smoothing factor (controls the balance between exploration and exploitation)
            num_perturbations: Number of perturbations to use for gradient estimation
            advantages: Advantages for policy gradient (used in on-policy methods)
            
        Returns:
            Smoothened gradient estimates, original gradients, and loss
        """
        # Flatten parameters for efficient operations
        flattened_params = self.flatten_parameters()
        
        # Initialize smoothened gradient accumulator 
        smoothened_gradient = torch.zeros_like(flattened_params)
        
        # Dynamic perturbation scale (smaller c for fine-grained estimation)
        c =  0.1  # Scale with beta as recommended
        
        # Get initial action probabilities
        action_probs = self(states)
        
        # Compute log probabilities of taken actions
        action_indices = actions
        log_probs = torch.log(torch.gather(action_probs, dim=1, index=action_indices))
        
        # Get value estimates or use provided advantages
        if advantages is None:
            values = critic(states, action_probs).detach()
            advantages = values
        
        # Calculate entropy for exploration bonus
        entropy = -torch.mean(torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=1))
        entropy_coef = 0.01  # Entropy coefficient
        
        # Compute policy loss with entropy bonus (negative for gradient ascent)
        original_loss = -torch.mean(advantages * log_probs) - entropy_coef * entropy
        original_loss.backward()

        # Flatten original gradients
        original_gradients = torch.cat([param.grad.view(-1) for param in self.parameters()])
        
        # Apply gradient clipping to original gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=[p for p in self.parameters()], 
            max_norm=0.5,  # As recommended
            norm_type=2
        )

        # Sample a random Bernoulli perturbation vector δ_k = ±1
        # Preserving your Bernoulli perturbation approach as it's core to SFAC
        delta_k = 2 * torch.bernoulli(torch.full_like(flattened_params, 0.5)) - 1
        
        # Clone parameters for perturbation
        original_params = [p.clone() for p in self.parameters()]
        
        # Perturb the parameters θ1 = θ + c * δ_k
        for i, param in enumerate(self.parameters()):

            start_idx = 0
            if i > 0:
                start_idx = sum(p.numel() for p in list(self.parameters())[:i])
            end_idx = start_idx + param.numel()
            param_delta = delta_k[start_idx:end_idx].view(param.shape)
            param.data.add_(c * param_delta)  # Apply positive perturbation
        
        # Compute gradients for θ1 using rollouts (preserving your approach)
        self.zero_grad()
        log_probs_1, rewards_1 = self.compute_rollout()
        loss_1 = -torch.mean(rewards_1 * log_probs_1)
        loss_1.backward()
        grad_theta1 = torch.cat([param.grad.view(-1) for param in self.parameters()])
        
        # Restore parameters and apply negative perturbation
        for i, (param, orig) in enumerate(zip(self.parameters(), original_params)):
            param.data.copy_(orig)
            start_idx = 0
            if i > 0:
                start_idx = sum(p.numel() for p in list(self.parameters())[:i])
            end_idx = start_idx + param.numel()
            param_delta = delta_k[start_idx:end_idx].view(param.shape)
            param.data.sub_(c * param_delta)  # Apply negative perturbation
            
        # Compute gradients for θ2 using rollouts
        self.zero_grad()
        log_probs_2, rewards_2 = self.compute_rollout()
        loss_2 = -torch.mean(rewards_2 * log_probs_2)
        loss_2.backward()
        grad_theta2 = torch.cat([param.grad.view(-1) for param in self.parameters()])
        
        # Restore original parameters
        for param, orig in zip(self.parameters(), original_params):
            param.data.copy_(orig)
            
        # Calculate gradient differences for second-order term
        delta_G_k = grad_theta1 - grad_theta2
        perturbation_buffer = torch.empty_like(flattened_params) 
        # Perform Gaussian perturbations to accumulate the smoothened gradient (SFAC core)
        for i in range(num_perturbations):
            # print("Perturbation iteration:", i)
            # Sample Gaussian perturbation as in your original code
            # rho_k = torch.randn_like(flattened_params)

            perturbation_buffer.normal_() 
            rho_k = perturbation_buffer 
            
            # First order term: β * ρ_k^T ∇θ J(θ) 
            rho_dot_grad = torch.dot(rho_k, original_gradients)
            
            # Second order terms as in your original code
            delta_k_dot_rho = torch.dot(delta_k, rho_k)  # (ρᵢᵀΔₖ)
            delta_g_dot_rho = torch.dot(delta_G_k, rho_k)  # (ρᵢᵀδGₖ)
            rho_dot_delta_g = torch.dot(rho_k, delta_G_k)  # Same as above
            delta_k_transpose_rho = torch.dot(delta_k, rho_k)  # (Δₖᵀρᵢ)

            # Second order term as in your implementation
            second_order_term = 0.5 * beta**2 * (
                delta_k_dot_rho * delta_g_dot_rho + 
                rho_dot_delta_g * delta_k_transpose_rho
            )

            # Update the smoothened gradient using your approach
            smoothened_gradient += rho_k * (original_loss + beta * rho_dot_grad + second_order_term)

        # Normalize by number of perturbations and beta
        smoothened_gradient = smoothened_gradient / (num_perturbations * beta)
        
        return smoothened_gradient, original_gradients, original_loss
        
    def compute_gradients(self, critic, theta):
        """
        Compute gradients for a given set of parameters θ using rollouts.
        """
        # Save original parameters and gradients
        original_params = [param.clone() for param in self.parameters()]
        original_grads = [param.grad.clone() if param.grad is not None else None for param in self.parameters()]
        
        # Set parameters to the provided theta values
        # Handle the case where theta is a flattened vector
        if isinstance(theta, torch.Tensor) and theta.dim() == 1:
            idx = 0
            for param in self.parameters():
                num_params = param.numel()
                param.data.copy_(theta[idx:idx+num_params].view(param.shape))
                idx += num_params
        else:
            # Assume theta is a list of parameter tensors
            for param, theta_param in zip(self.parameters(), theta):
                param.data.copy_(theta_param)

        # Zero out previous gradients
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()

        # Run rollouts with the new parameters
        log_probs, cumulative_rewards = self.compute_rollout()
        
        # Calculate loss and compute gradients
        loss = -torch.mean(cumulative_rewards * log_probs)
        loss.backward()

        # Collect gradients into a flattened vector
        gradients = torch.cat([param.grad.view(-1) for param in self.parameters()])

        # Restore original parameters and gradients
        for param, orig_param, orig_grad in zip(self.parameters(), original_params, original_grads):
            param.data.copy_(orig_param)
            if orig_grad is not None:
                param.grad = orig_grad

        return gradients
        
    def compute_rollout(self):
        """
        Run rollouts in the Assault-v5 environment and compute log-probabilities and discounted rewards.
        Uses preprocessing for better feature extraction.
        """
        env_temp = gym.make('ALE/Assault-v5', render_mode='rgb_array')
        preprocessor = AtariPreprocessor()  # Using our new preprocessor
        
        log_probs_list = []
        cumulative_rewards_list = []
        states_list = []
        actions_list = []
        
        # Run fewer episodes (3 instead of 5) as recommended for efficiency
        for _ in range(3):
            state, _ = env_temp.reset()
            state = preprocessor.process_state(state, reset=True)  # Preprocess initial state
            
            rewards = []
            done = False
            episode_states = []
            episode_actions = []
            
            # Run episode
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action_probs = self(state_tensor)
                action = self.sample_action(action_probs[0])  # Sample action
                
                next_state, reward, done, truncated, _ = env_temp.step(action)
                next_state = preprocessor.process_state(next_state)  # Preprocess
                reward = preprocessor.clip_reward(reward)  # Clip rewards as recommended
                
                # Store transition
                episode_states.append(state)
                episode_actions.append(action)
                rewards.append(reward)
                
                # Update state
                state = next_state
                if truncated:
                    done = True
                    
            # Add states and actions from this episode to our lists
            states_list.extend(episode_states)
            actions_list.extend(episode_actions)
            
            # Calculate discounted cumulative rewards
            discounted_reward = 0
            discounted_rewards = []
            for r in reversed(rewards):
                discounted_reward = r + GAMMA * discounted_reward
                discounted_rewards.insert(0, discounted_reward)
                
            # Store the cumulative (discounted) reward of the episode
            cumulative_rewards_list.append(discounted_rewards[0])
        print("closing the env")   
        env_temp.close()
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.stack(states_list)).to(device)
        actions_tensor = torch.LongTensor(actions_list).unsqueeze(1).to(device)
        
        # Get log probabilities of taken actions
        action_probs = self(states_tensor)
        log_probs = torch.log(torch.gather(action_probs, dim=1, index=actions_tensor))
        del states_tensor, actions_tensor, action_probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                
        # Convert rewards to tensor
        cumulative_rewards = torch.tensor(cumulative_rewards_list, dtype=torch.float32).to(device)
        
        return log_probs, cumulative_rewards