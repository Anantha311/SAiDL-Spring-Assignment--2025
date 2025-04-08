import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


# Step 1: Create the environment
env = gym.make('CartPole-v1',render_mode ='human')

# Step 2: Initialize DQN with custom exploration parameters
model = DQN(
    "MlpPolicy",               # Neural network architecture (default: 2-layer MLP)
    env, 
    learning_rate=1e-3,        # Learning rate for optimizer
    buffer_size=10000,         # Replay buffer size
    exploration_fraction=0.3,  # % of total steps for exploration
    exploration_initial_eps=1.0, # Start with full exploration
    exploration_final_eps=0.05,  # End with minimal exploration
    train_freq=4,              # Train every 4 steps
    target_update_interval=500, # Update target network every 500 steps
    verbose=1                  # Print training progress
)

# Step 3: Train the model
model.learn(total_timesteps=50000)

# Step 4: Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)

print(f"Average Reward: {mean_reward:.2f} ± {std_reward:.2f}")

# Step 5: Test the trained model
obs, _ = env.reset()
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
