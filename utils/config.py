"""
Global configuration parameters for SFAC algorithm
"""

# Environment settings
ENV_NAME = "ALE/Assault-v5"
SEED = 42

# Algorithm selection
ALGORITHM = "sfac"  # Options: "sfac", "pg" (policy gradient)

# Training parameters
NUM_EPISODES = 4000
MAX_STEPS_PER_EPISODE = 10000
GAMMA = 0.99

# Learning rates
LR_ACTOR = 0.0001
LR_CRITIC = 0.0001

# Network architecture
HIDDEN1_DIM = 512
HIDDEN2_DIM = 256

# SFAC parameters
BETA_INIT = 0.99
T_INIT = 100
T_MIN = 20
T_MAX = 100
TARGET_VARIANCE = 0.01

# Batch and buffer sizes
BATCH_SIZE = 2048
UPDATE_EVERY = 2048  # Update after collecting this many transitions
REWARD_BUFFER_SIZE = 100
REPLAY_SIZE = 1000000

# Gradient clipping
ACTOR_MAX_GRAD_NORM = 0.5
CRITIC_MAX_GRAD_NORM = 1.0

# Beta adjustment parameters
BETA_INCREASE_FACTOR = 1.1
BETA_DECREASE_FACTOR = 0.9
BETA_MIN = 0.3
BETA_MAX = 5.0

# T adjustment parameters
DELTA_T_UP = 100
DELTA_T_DOWN = 50

# GAE parameters
GAE_LAMBDA = 0.95

# Regularization
ENTROPY_COEF = 0.01
WEIGHT_DECAY = 1e-5

# Preprocessing parameters
FRAME_STACK = 4
FRAME_SIZE = (84, 84)
REWARD_CLIP = 1.0

# Evaluation parameters
EVAL_FREQUENCY = 100  # Episodes between evaluations
EVAL_EPISODES = 10  # Number of episodes for evaluation
RENDER_EVAL = False  # Whether to render evaluation episodes

# Checkpoint parameters
SAVE_FREQUENCY = 50  # Episodes between checkpoints
EARLY_STOPPING_PATIENCE = 20  # Number of evals with no improvement before stopping

# Add these to config.py
N_EPOCHS = 4  # Number of PPO-style epochs
MINIBATCH_SIZE = 512  # Size of minibatch for updates