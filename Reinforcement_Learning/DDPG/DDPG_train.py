import gymnasium as gym
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

# Create the Hopper-v4 environment
env = gym.make("Hopper-v4")

# Define action noise for exploration (ε ~ N in pseudocode)
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Initialize the DDPG model
model = DDPG(
    "MlpPolicy",         # Policy network
    env,                 # Environment
    action_noise=action_noise,
    learning_rate=1e-3,  # Optimizer learning rate
    buffer_size=1000000, # Replay buffer size
    learning_starts=1000,# Start learning after 1000 steps
    batch_size=64,       # Batch size for sampling
    gamma=0.99,          # Discount factor
    tau=0.005,           # Polyak averaging factor (ρ in pseudocode)
    verbose=1,           # Display training logs
    device='cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
)

# Train the model
model.learn(total_timesteps=500000, log_interval=10)

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean Reward: {mean_reward} ± {std_reward}")

# Save the trained model
model.save("ddpg_hopper_v4")

# Load the model and test it
model = DDPG.load("ddpg_hopper_v4")

# Test the trained agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()
