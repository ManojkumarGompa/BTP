import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from torchvision import transforms
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Hyperparameters
GAMMA = 0.9
TAU = 0.005
LR_ACTOR = 0.0001
LR_CRITIC = 0.001
MEMORY_SIZE = 10000
BATCH_SIZE = 64*2
UPDATE_EVERY = 4
NUM_EPISODES = 3000
RENDER_EVERY = 50
action_size=7

#setting seed : different seeds for different iterations
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# Preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),  # Convert to grayscale
        transforms.Resize((84, 84)),  # Resize to 84x84
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    return transform(image).unsqueeze(0).view(-1)

# Define the Actor Network
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)  # Use softmax for discrete actions

    # Helper function to perform forward pass with manually perturbed parameters
    # def forward_with_params(self, x, perturbed_params):
    #     x = torch.relu(nn.functional.linear(x, perturbed_params[0], perturbed_params[1]))
    #     x = torch.relu(nn.functional.linear(x, perturbed_params[2], perturbed_params[3]))
    #     return torch.softmax(nn.functional.linear(x, perturbed_params[4], perturbed_params[5]), dim=-1)

    def regular_gradient_update(self, states, actions, critic):
        # Compute action probabilities
        mean = self(states)
        action_indices = actions
        log_probs = torch.log(torch.gather(mean, dim=1, index=action_indices))

        # Compute the Q-value from the critic
        Q_values = critic(states, actions).detach()

        # Compute actor loss
        loss = -torch.mean(Q_values * log_probs)  # Regular policy gradient loss
        loss.backward()  # Backpropagate gradients

        return loss.item()
    def flatten_parameters(self):
        """
        Flatten the parameters from all layers into a single list.
        """
        params = []
        for param in self.parameters():
            params.append(param)
        return params

   
    def smoothened_gradient_update(self, states, actions, critic, beta, num_perturbations=5):
        smoothened_gradient = [torch.zeros_like(param) for param in self.flatten_parameters()]
        c = 0.01

        # Compute the log-probabilities of the actions from the policy
        action_probs = self(states)  # Get action probabilities
        action_indices = actions
        log_probs = torch.log(torch.gather(action_probs, dim=1, index=action_indices))  # Log-prob of taken actions

        # Compute the Q-value from the critic
        Q_values = critic(states, action_probs).detach()

        # Compute the actor loss: ∇J(θ) = E[Q(s,a) * ∇θ log πθ(a|s)]
        original_loss = -torch.mean(Q_values * log_probs)  # Maximize expected Q-value
        original_loss.backward(retain_graph=True)  # Retain graph for further computation

        # Compute the original gradient ∇θJ(θ)
        original_gradients = [param.grad.clone() for param in self.flatten_parameters()]
        
         # Sample a random Bernoulli perturbation vector δ_k = ±1
        # Generate a Bernoulli distribution with p=0.5 for each parameter's shape
         # If the result is 1 (heads), it's +1, if it's 0 (tails), it's -1
        delta_k = [2 * torch.bernoulli(torch.full(param.shape, 0.5)).float() - 1 for param in self.flatten_parameters()]

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
        delta_G_k = [g1 - g2 for g1, g2 in zip(grad_theta1, grad_theta2)]
        # Perform perturbations to accumulate the smoothened gradient
        for _ in range(num_perturbations):
           
           

            # Compute second-order term
            for i, (param, grad) in enumerate(zip(self.flatten_parameters(), original_gradients)):
                rho_k = torch.randn_like(param)  # Gaussian perturbation

                # First order: β * ρ_k^T ∇θ J(θ)
                rho_dot_grad = torch.dot(rho_k.view(-1), grad.view(-1))

                # Compute second-order terms
                

              
                delta_k_dot_rho = torch.dot(delta_k[i].view(-1), rho_k.view(-1))  # (ρᵢᵀΔₖ)
                delta_g_dot_rho = torch.dot(delta_G_k[i].view(-1), rho_k.view(-1))  # (ρᵢᵀδGₖ)
                rho_dot_delta_g = torch.dot(rho_k.view(-1), delta_G_k[i].view(-1))  # (ρᵢᵀδGₖ) same as above
                delta_k_transpose_rho = torch.dot(delta_k[i].view(-1), rho_k.view(-1))  # (Δₖᵀρᵢ)

                # Second order: β^2 / 2 [(ρᵢᵀΔₖ)(δGₖᵀρᵢ) + (ρᵢᵀδGₖ)(Δₖᵀρᵢ)]
                second_order_term = 0.5 * beta ** 2 * (
                    delta_k_dot_rho * delta_g_dot_rho + rho_dot_delta_g * delta_k_transpose_rho
                )
    

                # Update the smoothened gradient
                smoothened_gradient[i] += rho_k * (original_loss + beta * rho_dot_grad + second_order_term)

        # Normalize the smoothened gradient by the number of perturbations and beta
        smoothened_gradient = [sg / (num_perturbations * beta) for sg in smoothened_gradient]

        return smoothened_gradient


    # def compute_gradients(self, critic, theta):
    #     """
    #     Compute gradients for a given set of parameters θ using rollouts.
    #     """
    #     original_params = [param.clone() for param in self.flatten_parameters()]  # Save original parameters

    #     # Temporarily set parameters to θ
    #     print("shape of theta: ",len(theta),theta[0].shape)
        
    #     for param, theta_param in zip(self.flatten_parameters(), theta):
    #         param.data.copy_(theta_param)

    #     # Run rollouts and get log-probabilities and cumulative rewards
    #     log_probs, cumulative_rewards = self.compute_rollout()
        
    #     # Calculate loss using cumulative rewards and log-probabilities of initial actions
    #     loss = torch.mean(cumulative_rewards * log_probs)
    #     print("loss is ",loss,loss.shape)
    #     # Use autograd to compute gradients w.r.t the given parameters θ
    #     gradients = torch.autograd.grad(loss, theta, retain_graph=True)

    #     # Restore original parameters
    #     for param, original_param in zip(self.flatten_parameters(), original_params):
    #         param.data.copy_(original_param)

    #     return gradients
    def compute_gradients(self, critic, theta):
        """
        Compute gradients for a given set of parameters θ using rollouts.
        """
        original_params = [param.clone() for param in self.flatten_parameters()]  # Save original parameters
        original_grads = [param.grad.clone() if param.grad is not None else None for param in self.flatten_parameters()]  # Save original gradients

        # Temporarily set parameters to θ
        # print("shape of theta: ", len(theta), theta[0].shape)
        # print("shape of grads: ",len(original_grads),original_grads[0].shape)
        
        for param, theta_param in zip(self.flatten_parameters(), theta):
            param.data.copy_(theta_param)

        # Zero out previous gradients (this is important for a clean backward pass)
        for param in self.flatten_parameters():
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
        gradients = [param.grad.clone() for param in self.flatten_parameters()]
        # print(gradients)
        # Restore original parameters and gradients
        for param, original_param, original_grad in zip(self.flatten_parameters(), original_params, original_grads):
            param.data.copy_(original_param)  # Restore original parameter values
            if original_grad is not None:
                param.grad = original_grad  # Restore original gradients

        return gradients



    def select_action_actor_custom(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self(state).detach().numpy()[0]

        if np.any(np.isnan(action_probs)) or np.any(action_probs < 0):
            print("Warning: NaN or negative action probabilities detected. Resetting to uniform distribution.")
            action_probs = np.ones_like(action_probs) / len(action_probs)

        return np.random.choice(len(action_probs), p=action_probs)
    def compute_rollout(self):
        """
        Run rollouts in the `assault-v4` environment and compute log-probabilities and discounted cumulative rewards.
        Only considers the log-probability of the action taken at the start of the episode.
        
        Parameters:
        - gamma (float): Discount factor for rewards (default: 0.99)
        """
        env_temp = gym.make('Assault-v4', render_mode='rgb_array')
        log_probs = []
        cumulative_rewards = []

        for _ in range(3):  # Run 3 episodes
            state, _ = env_temp.reset()
            state = preprocess_image(state)  # Preprocess and convert to tensor

            # Get action probabilities for the initial state
            action = self.select_action_actor_custom(state)  # Select action based on initial state
            action_tensor = torch.tensor([action], dtype=torch.long)  # Convert action to 1D tensor
            action_probs = self(state)  # Action probabilities from the actor
            action_probs_tensor = torch.FloatTensor(action_probs).unsqueeze(0)  # Add batch dimension
            log_prob = torch.log(action_probs_tensor.gather(dim=1, index=action_tensor.unsqueeze(0)))  # Log-probability
            log_probs.append(log_prob.squeeze())  # Append as scalar (1D)

            rewards = []
            done = False

            while not done:
                next_state, reward, done, _, _ = env_temp.step(action)
                next_state = preprocess_image(next_state)  # Preprocess next state
                rewards.append(reward)  # Collect all rewards in the episode
                action = self.select_action_actor_custom(next_state)

            # Calculate discounted cumulative rewards
            discounted_reward = 0
            discounted_rewards = []
            for r in reversed(rewards):
                discounted_reward = r + GAMMA * discounted_reward  # Apply discount factor
                discounted_rewards.insert(0, discounted_reward)  # Insert at the beginning

            cumulative_rewards.append(discounted_rewards[0])  # Store total discounted reward for the episode

        env_temp.close()

        # Convert to tensors with consistent shapes
        log_probs = torch.stack(log_probs)  # Shape: (5,)
        cumulative_rewards = torch.tensor(cumulative_rewards, dtype=torch.float32)  # Shape: (5,)

        return log_probs, cumulative_rewards


# Define the Critic Network (Q-value function)
class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 128)  # Input is state + one-hot action
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)  # Output is a single Q-value

    def forward(self, state, action):
        # Concatenate state and one-hot action
        x = torch.cat([state, action], dim=1)  # Concatenate along the feature dimension
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Q-value as the output


# Experience Replay Memory
class ReplayBuffer:
    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def size(self):
        return len(self.memory)


# DDPG Agent
class DDPGAgent:
    def __init__(self, state_size, action_size):
        self.actor = Actor(state_size, action_size).float()
        self.critic = Critic(state_size, action_size).float()
        self.target_actor = Actor(state_size, action_size).float()
        self.target_critic = Critic(state_size, action_size).float()

        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.update_target_networks(tau=1.0)

    def update_target_networks(self, tau=TAU):
        # Update the target networks using soft updates
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state).detach().numpy()[0]

        if np.any(np.isnan(action_probs)) or np.any(action_probs < 0):
            print("Warning: NaN or negative action probabilities detected. Resetting to uniform distribution.")
            action_probs = np.ones_like(action_probs) / len(action_probs)

        return np.random.choice(len(action_probs), p=action_probs)

    def update(self, use_smoothened_gradient=True,num_perturbations=500):
        if self.memory.size() < BATCH_SIZE:
            return 0  # Return a dummy loss
    
        transitions = self.memory.sample(BATCH_SIZE)
        batch = list(zip(*transitions))
    
        states = torch.FloatTensor(np.stack(batch[0]))
        actions = torch.LongTensor(batch[1]).unsqueeze(1)
        rewards = torch.FloatTensor(batch[2]).unsqueeze(1)
        next_states = torch.FloatTensor(np.stack(batch[3]))
        dones = torch.FloatTensor(batch[4]).unsqueeze(1)
    
        actions_one_hot = torch.zeros(BATCH_SIZE, action_size).scatter(1, actions, 1)
    
        with torch.no_grad():
            target_next_actions = self.target_actor(next_states)
            target_q_values = rewards + (1 - dones) * GAMMA * self.target_critic(next_states, target_next_actions)
    
        current_q_values = self.critic(states, actions_one_hot)
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = 0
        if use_smoothened_gradient:
            self.actor_optimizer.zero_grad()
            smoothened_gradient = self.actor.smoothened_gradient_update(states, actions, self.critic, beta=2,num_perturbations=num_perturbations)
            with torch.no_grad():
                for param, grad in zip(self.actor.parameters(), smoothened_gradient):
                    param.grad = grad
            self.actor_optimizer.step()
        else:
            self.actor_optimizer.zero_grad()
            actor_loss = self.actor.regular_gradient_update(states, actions, self.critic)
            self.actor_optimizer.step()
    
        self.update_target_networks()
        return actor_loss


# Training and Testing
def train_ddpg(agent, env, use_smoothened_gradient=True,num_perturbations=500):
    scores = []
    actor_losses = []
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        state = preprocess_image(state)
        score = 0
        done = False
        

        while not done:
            
            action = agent.select_action(state)
            next_state, reward, done, info, _ = env.step(action)
            next_state = preprocess_image(next_state)
            agent.memory.push((state, action, reward, next_state, float(done)))
            
            state = next_state
            score += reward

        scores.append(score)
        loss = agent.update(use_smoothened_gradient,num_perturbations=num_perturbations)
        actor_losses.append(loss)
        print(f'Episode {episode}: Score: {score}')

    return scores, actor_losses

def visualize_performance(agent, env_name, num_episodes=1):
    env = gym.make(env_name, render_mode='human')  # Use 'human' render mode for visualization
    
    for i in range(num_episodes):
        state, _ = env.reset()
        state = preprocess_image(state)  # Preprocess the state
        done = False
        while not done:
            action = agent.select_action(state)  # No exploration noise
            next_state, reward, done, info, _ = env.step(action)  # Correct unpacking
            state = preprocess_image(next_state)  # Preprocess the next state
    env.close()
# Main program

def polyak_averaging(scores, alpha=0.99):
    smoothed_scores = []
    smoothed_value = scores[0]  # Initialize with the first score
    for score in scores:
        smoothed_value = alpha * smoothed_value + (1 - alpha) * score
        smoothed_scores.append(smoothed_value)
    return smoothed_scores




def main():
    print("Starting the program")
    env = gym.make('Assault-v4', render_mode='rgb_array')
    state_size = 84 * 84
    num_runs = 3  # Number of times to run the training
    num_perturbations_list = [2000]  # Different perturbation values to test

    for num_perturbations in num_perturbations_list:
        print(f"Running experiments with NUM_PERTURBATIONS = {num_perturbations}")
        all_smoothened_scores = []
        all_regular_scores = []

        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs} for NUM_PERTURBATIONS = {num_perturbations}")
            random_seed = random.randint(0, 10000)
            set_seed(random_seed)

            # Initialize agent
            agent = DDPGAgent(state_size, 7)

            # Save the initial actor and critic parameters
            initial_actor_weights = agent.actor.state_dict()
            initial_critic_weights = agent.critic.state_dict()

            # Train using smoothened gradient
            print("Training with smoothened gradient...")
            smoothened_scores, _ = train_ddpg(agent, env, use_smoothened_gradient=True, num_perturbations=num_perturbations)
            all_smoothened_scores.append(smoothened_scores)

            # Reinitialize agent to avoid interference, but restore initial weights
            agent = DDPGAgent(state_size, 7)
            agent.actor.load_state_dict(initial_actor_weights)
            agent.critic.load_state_dict(initial_critic_weights)

            # Train using regular policy gradient
            set_seed(random_seed)
            print("Training with regular gradient...")
            regular_scores, _ = train_ddpg(agent, env, use_smoothened_gradient=False)
            all_regular_scores.append(regular_scores)

            # Save the comparison plot for this run
            plt.figure()
            # Apply Polyak averaging to the scores of the current run
            polyak_smoothened_scores = polyak_averaging(smoothened_scores, alpha=0.99)
            polyak_regular_scores = polyak_averaging(regular_scores, alpha=0.99)

            # Plot the Polyak-averaged scores
            plt.plot(polyak_smoothened_scores, label=f'Smoothened Gradient Run {run + 1}', color='b')
            plt.plot(polyak_regular_scores, label=f'Regular Gradient Run {run + 1}', color='g')

            # plt.plot(smoothened_scores, label=f'Smoothened Gradient Run {run + 1}', color='b')
            # plt.plot(regular_scores, label=f'Regular Gradient Run {run + 1}', color='g')
            plt.title(f"Comparison of Gradients (Run {run + 1})")
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.legend()
            plt.savefig(f'comparison_run_{run + 1}_perturbations_{num_perturbations}.png')  # Save comparison plot for this run
            plt.close()  # Close the plot after saving it to avoid memory issues

        # Average scores over all runs
        averaged_smoothened_scores = np.mean(all_smoothened_scores, axis=0)
        averaged_regular_scores = np.mean(all_regular_scores, axis=0)

        # Smooth both score curves using Polyak averaging
        smoothed_smoothened_scores = polyak_averaging(averaged_smoothened_scores, alpha=0.99)
        smoothed_regular_scores = polyak_averaging(averaged_regular_scores, alpha=0.99)

        # Plot the averaged comparison scores after all runs
        plt.figure()
        plt.plot(smoothed_smoothened_scores, label=f'Smoothened Gradient (NUM_PERTURBATIONS = {num_perturbations})', color='b')
        plt.plot(smoothed_regular_scores, label=f'Regular Gradient (NUM_PERTURBATIONS = {num_perturbations})', color='g')
        plt.title(f"Performance Comparison with NUM_PERTURBATIONS = {num_perturbations}")
        plt.ylabel('Score')
        plt.xlabel('Episode')
        plt.legend()
        plt.savefig(f'comparison_avg_plot_perturbations_{num_perturbations}.png')  # Save the averaged comparison plot
        plt.show()

if __name__ == '__main__':
    main()

