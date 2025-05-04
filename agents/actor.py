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
env = gym.make("ALE/Assault-v5", render_mode='rgb_array')

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
        # epsilon = 0.1  # Exploration rate

        # if random.random() < epsilon:
        #     # Uniformly random action from 0 to 6
        #     action = torch.randint(0, 7, (1,)).item()
        #     return action

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

    # def flatten_parameters(self):
    #     params = []
    #     for param in self.parameters():
    #         params.append(param)
    #     return params
    def flatten_parameters(self):
        """
        Flatten all parameters into a single vector.
        """
        return torch.cat([param.view(-1) for param in self.parameters()])

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
        flattened_params=self.flatten_parameters()
        smoothened_gradient = torch.zeros_like(flattened_params)
        
        
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
        # original_gradients = [param.grad.clone() for param in self.flatten_parameters()]
         # Flatten original gradients
        original_gradients = torch.cat([param.grad.view(-1) for param in self.parameters()])

          # Sample a random Bernoulli perturbation vector δ_k = ±1
        # Generate a Bernoulli distribution with p=0.5 for each parameter's shape
         # If the result is 1 (heads), it's +1, if it's 0 (tails), it's -1
        delta_k = 2 * torch.bernoulli(torch.full(flattened_params.shape, 0.5)).float() - 1

        # delta_k = [torch.randint(0, 2, param.shape).float() * 2 - 1 for param in self.flatten_parameters()]  # +1 or -1

        # Perturb the parameters θ1 = θ + c * δ_k, θ2 = θ - c * δ_k
        theta1 = [param + c * delta_k_i for param, delta_k_i in zip(self.flatten_parameters(), delta_k)]
        theta2 = [param - c * delta_k_i for param, delta_k_i in zip(self.flatten_parameters(), delta_k)]

        # Ensure that these new tensors have requires_grad set to True
        theta1 = [param.clone().detach().requires_grad_(True) for param in theta1]
        theta2 = [param.clone().detach().requires_grad_(True) for param in theta2]

        # Compute gradients using rollouts for θ1 and θ2
        grad_theta1 = self.compute_gradients(critic, theta1)
        grad_theta2 = self.compute_gradients(critic, theta2)

        # Compute the difference in gradients: δG_k = ∇θ1 L(θ1) - ∇θ2 L(θ2)
        delta_G_k = grad_theta1 - grad_theta2
        # Perform perturbations to accumulate the smoothened gradient
        for _ in range(num_perturbations):

                rho_k = torch.randn_like(flattened_params)  # Gaussian perturbation

                # First order: β * ρ_k^T ∇θ J(θ)
                rho_dot_grad = torch.dot(rho_k.view(-1),original_gradients)

                # Compute second-order terms
                

              
                delta_k_dot_rho = torch.dot(delta_k, rho_k)  # (ρᵢᵀΔₖ)
                delta_g_dot_rho = torch.dot(delta_G_k, rho_k)  # (ρᵢᵀδGₖ)
                rho_dot_delta_g = torch.dot(rho_k, delta_G_k)  # (ρᵢᵀδGₖ) same as above
                delta_k_transpose_rho = torch.dot(delta_k, rho_k)  # (Δₖᵀρᵢ)

                # Second order: β^2 / 2 [(ρᵢᵀΔₖ)(δGₖᵀρᵢ) + (ρᵢᵀδGₖ)(Δₖᵀρᵢ)]
                second_order_term = 0.5 * beta ** 2 * (
                    delta_k_dot_rho * delta_g_dot_rho + rho_dot_delta_g * delta_k_transpose_rho
                )
    

                # Update the smoothened gradient
                smoothened_gradient+= rho_k * (original_loss + beta * rho_dot_grad + second_order_term)

        # Normalize the smoothened gradient by the number of perturbations and beta
        smoothened_gradient = [sg / (num_perturbations * beta) for sg in smoothened_gradient]
        return smoothened_gradient,original_gradients,original_loss
    
    def compute_gradients(self, critic, theta):
        """
        Compute gradients for a given set of parameters θ using rollouts.
        """
        original_params = [param.clone() for param in self.parameters()]  # Save original parameters
        original_grads = [param.grad.clone() if param.grad is not None else None for param in self.parameters()]  # Save original gradients


        
        for param, theta_param in zip(self.parameters(), theta):
            param.data.copy_(theta_param)

        # Zero out previous gradients (this is important for a clean backward pass)
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()

        # Run rollouts and get log-probabilities and cumulative rewards
        log_probs, cumulative_rewards = self.compute_rollout()
        
        # Calculate loss using cumulative rewards and log-probabilities of initial actions
        loss = -torch.mean(cumulative_rewards * log_probs)
        # print("loss is ", loss, loss.shape)
        
        # Perform backward pass to compute gradients
        loss.backward()

        # Store gradients after backward pass
        # Flatten gradients into a single vector
        gradients = torch.cat([param.grad.view(-1) for param in self.parameters()])

        # print(gradients)
        # Restore original parameters and gradients
        for param, original_param,original_grad in zip(self.parameters(), original_params,original_grads):
            param.data.copy_(original_param)  # Restore original 
            if original_grad is not None:
                param.grad = original_grad

        return gradients
    def compute_rollout(self):
        """
        Run rollouts in the `assault-v4` environment and compute log-probabilities and discounted cumulative rewards.
        Only considers the log-probability of the action taken at the start of the episode.
        
        Parameters:
        - gamma (float): Discount factor for rewards (default: 0.99)
        """
        env_temp = gym.make('ALE/Assault-v5', render_mode='rgb_array')
        log_probs = []
        cumulative_rewards = []
        states=[]
        actions=[]

        for _ in range(5):  # Run 5 episodes
            state, _ = env_temp.reset()
            

            rewards = []
            done = False

            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action_probs = self(state_tensor)
                action = self.sample_action(action_probs[0])  # Sample action
                next_state, reward, done, truncated, _ = env_temp.step(action)
                states.append(state)
                actions.append(action)

                state= next_state  # Update state
                rewards.append(reward)  # Collect all rewards in the episode

            # Calculate discounted cumulative rewards
            discounted_reward = 0
            discounted_rewards = []
            for r in reversed(rewards):
                discounted_reward = r + GAMMA * discounted_reward  # Apply discount factor
                discounted_rewards.insert(0, discounted_reward)  # Insert at the beginning

            cumulative_rewards.append(discounted_rewards[0])  # Store total discounted reward for the episode

        env_temp.close()
        states_tensor=torch.FloatTensor(np.stack(states))
        actions_tensor=torch.LongTensor(actions).unsqueeze(1)
        action_probs = self(states_tensor)
        action_indices = actions_tensor
        log_probs = torch.log(torch.gather(action_probs, dim=1, index=action_indices))

        # Convert to tensors with consistent shapes
       
        cumulative_rewards = torch.tensor(cumulative_rewards, dtype=torch.float32)  # Shape: (5,)

        return log_probs, cumulative_rewards


 

