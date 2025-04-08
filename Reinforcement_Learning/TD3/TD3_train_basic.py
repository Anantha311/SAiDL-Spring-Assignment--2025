import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module): # Making a Linear neural network which has noise
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential( # Neural Network of actor
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, x):
        return self.max_action * self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1) # Concatenates along the specified dimension
        return self.q1(sa), self.q2(sa) # The sa (which is state action pair) gets passed into the neural network


class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)

        # Target Networks
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        # Copy weights to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def select_action(self, state):
        with torch.no_grad(): 
            state = torch.FloatTensor(state).to(device)
            return self.actor(state).cpu().numpy()


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size

    def add(self, experience):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)  # Remove oldest experience if full
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


env = gym.make("Hopper-v4")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])  # Maximum value of action possible in the environment for Hopper-v4 is 1
agent = TD3Agent(state_dim, action_dim, max_action)

initial_noise = 1.5 * max_action  # Start with maximum noise
exploration_decay = 0.9995             # Gradual decay for stability
min_noise = 0.1 * max_action          # Ensure some exploration continues
num_episodes = 2500
gamma = 0.99
tau = 0.005
replay_buffer = ReplayBuffer(max_size=100000)
total_steps = 0 
reward_history = []

# TD3 Training Loop
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        # Add Gaussian noise for exploration
        noise = max(min_noise, initial_noise * (exploration_decay ** (total_steps // 100))) # initially exploring then exploiting
        
        action = agent.select_action(state) 
        action = action + np.random.normal(0, noise , size=action_dim)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_steps += 1

        # Store experience
        replay_buffer.add((state, action, reward, next_state, done))

        # Sample batch from buffer and update critics
        if len(replay_buffer.buffer) >= 64:
            batch = replay_buffer.sample(64)
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert to tensors and move to GPU
            states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
            actions = torch.tensor(np.array(actions), dtype=torch.float32, device=device)
            rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=device).unsqueeze(1)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
            dones = torch.tensor(np.array(dones), dtype=torch.float32, device=device).unsqueeze(1)

            # Compute target Q-value
            with torch.no_grad():
                target_actions = agent.actor_target(next_states)
                noise_tensor = torch.normal(0, 0.2, size=target_actions.shape, device=device)
                target_actions = target_actions + torch.clamp(noise_tensor, -0.5, 0.5)  # Add noise with limits to stabilize targets
                target_actions = torch.clamp(target_actions, -max_action, max_action)
                q1_target, q2_target = agent.critic_target(next_states, target_actions)
                min_Q = torch.min(q1_target, q2_target)
                target_value = rewards + gamma * (1 - dones) * min_Q

            # Update Critic
            q1, q2 = agent.critic(states, actions)  # Main critic output
            critic_loss = torch.mean(torch.square(q1 - target_value) + 
                                     torch.square(q2 - target_value))

            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()

            # Delayed Actor Update
            if episode % 2 == 0:
                actor_loss = -torch.mean(agent.critic(states, agent.actor(states))[0])
                agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                agent.actor_optimizer.step()

                # Polyak Averaging
                with torch.no_grad():
                    for param, target_param in zip(agent.critic.parameters(), agent.critic_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                    for param, target_param in zip(agent.actor.parameters(), agent.actor_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        state = next_state
        episode_reward += reward
    reward_history.append(episode_reward)
    
    if episode % 100 == 0:
        avg_reward = np.mean(reward_history[-100:]) # np.mean(reward_history[-100:]) calculates the average (mean) of the last 100 elements in the reward_history list.
        print(f"Episode {episode + 1}, Avg Reward: {avg_reward}, total Steps:{total_steps}")
        print(f"Current Noise Level: {noise:.2f}")

# Plot learning progress
plt.plot(reward_history)
plt.title("TD3 Learning Progress")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

# Save model
torch.save({
    'actor': agent.actor.state_dict(),
    'critic': agent.critic.state_dict(),
    'actor_target': agent.actor_target.state_dict(),
    'critic_target': agent.critic_target.state_dict(),
    'actor_optimizer': agent.actor_optimizer.state_dict(),
    'critic_optimizer': agent.critic_optimizer.state_dict(),
}, "td3_checkpoint.pth")
