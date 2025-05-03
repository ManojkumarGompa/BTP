from utils.config import NUM_EPISODES
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_ddpg(agent, env, use_smoothened_gradient=True, num_perturbations=500):
    scores = []
    actor_losses = []
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        state = torch.Tensor(state).to(device)
        mean=agent.actor(state)
        score = 0
        done = False

        while not done:
            action = agent.actor.sample_action(mean)
            # print the action
            # print(f"Action by seledct action method {action[0]}")
            # print(f"Action by seledct action method2 {action[0]}",type(action[0]))
            # print the action
            # print(f"Action by seledct action method3 {action}")
            next_state, reward, done, info, _ = env.step(action)
            agent.memory.push((state, action, reward, next_state, float(done)))

            state = next_state
            score += reward

        scores.append(score)
        loss = agent.update(use_smoothened_gradient, num_perturbations=num_perturbations)
        actor_losses.append(loss)
        print(f'Episode {episode}: Score: {score}')

    return scores, actor_losses