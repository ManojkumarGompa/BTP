import cv2
import numpy as np
from collections import deque
import torch

class AtariPreprocessor:
    def __init__(self, frame_stack=4, frame_size=(84, 84), device="cpu"):
        self.frame_stack = frame_stack
        self.frame_size = frame_size
        self.frames = deque(maxlen=frame_stack)
        self.device = device
        self.reset()
        
    def reset(self):
        """Reset frame buffer with zero frames"""
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(np.zeros(self.frame_size, dtype=np.float32))
            
    def process_frame(self, frame):
        """Convert RGB to grayscale, resize, normalize to [0,1]"""
        if len(frame.shape) == 3 and frame.shape[2] == 3:  # If RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize frame
        frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        return frame
        
    def process_state(self, state, reset=False):
        """Process frame and return stacked state"""
        if reset:
            self.reset()
            
        processed_frame = self.process_frame(state)
        self.frames.append(processed_frame)
        
        # Stack frames along first dimension for network input
        stacked_frames = np.stack(list(self.frames), axis=0)
        return stacked_frames
        
    def clip_reward(self, reward):
        # return np.clip(reward, -1.0, 1.0)
        return reward/10.0
        
    def to_tensor(self, state):
        """Convert state to tensor"""
        if isinstance(state, np.ndarray):
            return torch.FloatTensor(state).to(self.device)
        return state