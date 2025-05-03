import gymnasium as gym
import numpy as np
import random
import torch
from agents.sfac_agent import DDPGAgent
from utils.helpers import set_seed
from train.train_ddpg import train_ddpg
from train.visualize import visualize_performance
from utils.helpers import polyak_averaging
import matplotlib.pyplot as plt


import ale_py

gym.register_envs(ale_py)
from utils.config import NUM_RUNS
def main():
    print("Starting the program")
    # for i in gym.envs.registry.keys():
    #     print(i)
    env = gym.make('ALE/Assault-v5', render_mode='rgb_array')
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    # state_size = 84 * 84
    num_runs = NUM_RUNS  # Number of times to run the training
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
            agent = DDPGAgent(state_dim, action_dim)

            # Save the initial actor and critic parameters
            initial_actor_weights = agent.actor.state_dict()
            initial_critic_weights = agent.critic.state_dict()

            # Train using smoothened gradient
            print("Training with smoothened gradient...")
            smoothened_scores, _ = train_ddpg(agent, env, use_smoothened_gradient=True, num_perturbations=num_perturbations)
            all_smoothened_scores.append(smoothened_scores)

            # Reinitialize agent to avoid interference, but restore initial weights
            agent = DDPGAgent(state_dim, action_dim)
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

