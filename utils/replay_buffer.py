import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer"""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample a batch of experiences from the replay buffer"""
        if self.size() < batch_size:
            return random.sample(self.buffer, self.size())
        else:
            return random.sample(self.buffer, batch_size)
    
    def size(self):
        """Return the current size of the replay buffer"""
        return len(self.buffer)