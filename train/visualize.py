from utils.preprocessing import preprocess_image
import gymnasium as gym
def visualize_performance(agent, env_name, num_episodes=1):
    env = gym.make(env_name, render_mode='human')
    
    for i in range(num_episodes):
        state, _ = env.reset()
        state = preprocess_image(state)
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info, _ = env.step(action)
            state = preprocess_image(next_state)
    env.close()
