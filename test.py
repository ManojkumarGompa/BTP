import gymnasium as gym

all_envs = {env.id: env for env in gym.envs.registry.values()}
print("All environments registered:")
for key, value in all_envs.items():
    print(f"Key: {key}, Value: {value}")
