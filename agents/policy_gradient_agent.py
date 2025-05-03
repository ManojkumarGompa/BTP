from .actor import Actor
from .critic import Critic
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import config as cfg
from collections import deque

# Configuration parameters
GAMMA = cfg.GAMMA
TAU = cfg.TAU
LR_ACTOR = cfg.LR_ACTOR
LR_CRITIC = cfg.LR_CRITIC
action_size = cfg.action_size
NUM_TRAJECTORIES = 5  # Number of trajectories to collect before each update

class PolicyGradientAgent:
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
        self.trajectory_rewards = []
        
        # Initialize target networks with current parameters
        self.update_target_networks(tau=1.0)
        
    def update_target_networks(self, tau=TAU):
        """Soft update target networks"""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            
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
    
    def has_enough_trajectories(self):
        """Check if enough trajectories have been collected for an update"""
        return self.trajectory_count >= NUM_TRAJECTORIES
    
    def update(self):
        """Update policy and value functions using regular policy gradient"""
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

        # ----- Update Actor using Regular Policy Gradient -----
        self.actor_optimizer.zero_grad()
        
        # Get action probabilities
        action_probs = self.actor(states)
        
        # Get log probabilities of actions taken
        log_probs = torch.log(torch.gather(action_probs, 1, actions))
        
        # Get advantage estimates using critic
        with torch.no_grad():
            state_values = self.critic(states, action_probs)
            advantages = target_q_values - state_values
        
        # Compute policy gradient loss (negative for gradient ascent)
        actor_loss = -torch.mean(log_probs * advantages)
        
        # Update actor
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self.update_target_networks()
        
        return critic_loss.item()