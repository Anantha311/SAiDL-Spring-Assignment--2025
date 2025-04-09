
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NoisyLinear(nn.Module): # Making a Linear neural network which has noise
    def __init__(self, in_features, out_features, std_init=0.1,initial_decay_noise = 1 ):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features # This is number of input parameter
        self.out_features = out_features # This is number of input parameter
        self.std_init = std_init / np.sqrt(in_features) # Standard Deviation which controls the standard deviation of the noise 
        self.decay_noise = initial_decay_noise
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features)) #Creates an uninitialized tensor of shape (out_features, in_features), nn.Parameter(...) Converts the tensor into a trainable parameter so it is updated during training
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features)) # Same as above
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features)) # This creates and registers a non-trainable tensor (buffer) inside the PyTorch module. #Buffers are saved with the model but not updated during backpropagation. In NoisyLinear, buffers store random noise tensors for exploration
        self.min_noise = 0.1
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()
        
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight_mu, gain=0.1)  # Smaller initialization
        nn.init.uniform_(self.bias_mu, -0.01, 0.01)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))

    def reset_noise(self):
        # Truncated normal noise with controlled variance
        weight_noise = torch.randn_like(self.weight_epsilon) * 0.1 # Reduced scale
        self.weight_epsilon.data = weight_noise.clamp(-1.0, 1.0)  # Hard clipping
    
        bias_noise = torch.randn_like(self.bias_epsilon) * 0.1
        self.bias_epsilon.data = bias_noise.clamp(-0.5, 0.5)
    def forward(self, x):
        if self.training:
            #decay = decay = max(0.1, 1.0 - (100 / 1e5)) 
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon.detach()
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon.detach()
            # .detach is used when you need tensor values without affecting gradients
            #Before: weight_epsilon is part of the computation graph (gradients would flow through it).
            #After: weight_epsilon.detach() is treated as a constant during backpropagation
            
        else: # During testing mu is used, remember mu is the learnd component and epsilon(register_buffer) and sigma multipied together is the noise component 
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias) # performs a linear transformation (matrix multiplication + bias addition) on the input x using the noisy weights and biases.    



class Actor(nn.Module): # In hopper-v4 state_dim = 11  action_dim = 3 what are they search in internet
    def __init__(self, state_dim, action_dim, max_action): 
        super().__init__()
        self.net = nn.Sequential( # Neural Network of actor, It is a neural netwrk which contains weights such that your input will of states will give give outputs of action best to get the best reward
            NoisyLinear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.LayerNorm(256),
            NoisyLinear(256, 256),
            nn.ReLU(),
            NoisyLinear(256, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, x):
        return self.max_action * self.net(x)
    #self.net(x): Passes input x through your neural network (NoisyLinear layers + activations).
    #self.max_action * ...: Outputs raw action values in range [-1, 1] (due to final nn.Tanh()). Scales the [-1, 1] output to [-max_action, max_action].
   
    def reset_noise(self):
        for layer in self.net:
            if isinstance(layer,NoisyLinear):
                layer.reset_noise()

        # If the layer is NoisyLinear then reset the noise

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__() # A neural network which takes state_dim + action_dim and modifies weights such that it outputs the right reward
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1) # Concatenates along the specified dimension
        return self.q1(sa), self.q2(sa) # The sa (which is state action pair) gets passed into the neural network
    
class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action):

        self.actor = Actor(state_dim, action_dim, max_action).to(device) # Main Actor network of TD3 Agent
        self.critic = Critic(state_dim, action_dim).to(device) # Main critic network of TD3 Agent

        # Target Networks
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device) # Target Actor network of TD3 Agent
        self.critic_target = Critic(state_dim, action_dim).to(device) # Target Critic network of TD3 Agent

        # Optimizers
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=3e-5) # Optimizer of actor
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=3e-5)# Optimizer of critic
        # Copy weights to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.max_action = max_action
        self.total_steps = 0
        self.min_noise = 0.05
        self.initial_noise = 1
        self.exploration_decay = 0.9995
        self.noise = self.initial_noise
    def select_action(self, state, total_steps,exploration=True):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action = self.actor(state).cpu().numpy()
            
            if exploration:
                
                # #Progressively decaying noise
                # progress = min(1.0, self.total_steps / self.noise_schedule['decay_steps'])
                # noise_scale = self.noise_schedule['initial'] * (1 - progress) + self.noise_schedule['final'] * progress
                self.noise =  self.initial_noise * (self.exploration_decay ** (total_steps // 100))
                noise_scale = max(self.min_noise ,self.noise)
                action = action +  np.random.normal(0, noise_scale, size=action.shape)
                self.actor.reset_noise()
            
            return np.clip(action, -self.max_action, self.max_action)
    def update_noise(self):
        progress = min(1.0, self.total_steps / 1e5)
        for layer in self.actor.net:
            if isinstance(layer, NoisyLinear):
                layer.weight_sigma.data *= (1 - progress)  # Decay noise over time

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.4, beta=0.4,n_step=3, gamma=0.99,beta_increment=0.0001,device='cuda'):
        self.capacity = capacity # capacity: Maximum number of experiences the buffer can store.
        self.buffer = [] # buffer: Stores experiences (state, action, reward, next state, done).
        self.priorities = torch.zeros(capacity, dtype=torch.float32, device=device)  # Store priorities of experiences (important for sampling).
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta  # Importance sampling exponent
        self.beta_increment = beta_increment 
        self.pos = 0
        self.device = device
        self.n_step =n_step
        self.n_step_buffer = deque(maxlen=n_step) # because of dequeIf a new transition is added and the buffer is full, the oldest transition is automatically removed. 
        self.gamma = gamma

    def add(self, transition, td_error=1.0): # This doesnt calculate for last 2 states before it ends
        # transition: Experience tuple (state, action, reward, next_state, done).
        # td_error: The Temporal Difference (TD) error, used as priority.
        # Also we are doing n step returns for reward where n is 3 by default
        self.n_step_buffer.append(transition)
        if (len(self.n_step_buffer) >= self.n_step) or transition[-1]: # Either length of buffer greater or equal to n_step or the current trasition has done as true then:
            cumulative_reward = sum((self.gamma**i) * (self.n_step_buffer[i][2]) for i in range(self.n_step))
            n_step_state, n_step_action, _, _, _ = self.n_step_buffer[0]  # Get first transition
            _, _, _, n_step_next_state, n_step_done = self.n_step_buffer[-1]  # Get last transition
            # Store (state, action, n-step reward, next_state_N, done_N)
            full_transition = (n_step_state, n_step_action, cumulative_reward, n_step_next_state, n_step_done)
        else:
            full_transition = transition

        max_priority = self.priorities.max().item()  if len(self.buffer) > 0 else td_error # Gets the maximum item present in self.priorities.max().item() or gets td_error if length is zero -> using this logic till first capacity is not reached max_element will be td_error
        if len(self.buffer) < self.capacity:
            self.buffer.append(full_transition)
        else:
            self.buffer[self.pos] = full_transition
            #If buffer is not full, add new experience.
            #If full, overwrite the oldest experience(Oldest Experience at 0 which is self.pos).
        
        self.priorities[self.pos] = max_priority # adds the max_priority to self.priorities at self.pos
        # Adds the priority at self.pos
        self.pos = (self.pos + 1) % self.capacity  # Circular buffer
        # Goes on add self.pos till it reaches capacity then comes as 0 and adds then at 1 it adds (keep on replacing old experiences)


    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        # Common tensor conversion function
        def to_tensors(data):
            states, actions, rewards, next_states, dones = zip(*data)
            return (
                torch.FloatTensor(np.array(states)).to(self.device),
                torch.FloatTensor(np.array(actions)).to(self.device),
                torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device),
                torch.FloatTensor(np.array(next_states)).to(self.device),
                torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
            )

        if len(self.buffer) < self.capacity:  # Uniform sampling phase
            # Generate random indices
            indices = torch.randint(0, len(self.buffer), (batch_size,), device=self.device)
            samples = [self.buffer[i] for i in indices.cpu().numpy()]
            
            # Create dummy weights and indices for compatibility
            weights = torch.ones(batch_size, device=self.device)
            
        else:  # Prioritized sampling phase
            probs = (self.priorities[:len(self.buffer)] ** self.alpha) 
            probs /= probs.sum()
            
            indices = torch.multinomial(probs, batch_size, replacement=True)
            samples = [self.buffer[i] for i in indices.cpu().numpy()]
            
            weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
            weights /= weights.max()
            self.beta = min(1.0, self.beta + self.beta_increment)
            # print(f'weights: {weights}')
            # print(f'priority: {self.priorities}')
            # print(f'probs:{probs}')

        # Convert samples to tensors
        states, actions, rewards, next_states, dones = to_tensors(samples)
        
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        """Update priorities (ensure no inplace ops)"""
        new_priorities = torch.abs(td_errors.squeeze()) + 1e-5
        self.priorities[indices] = new_priorities  # Safe (not inplace)  # Avoid zero priority
            # We can do matrix addition even if indices has different indices from random sampling as pytorch takes care of it
        return self.priorities

env = gym.make("Hopper-v4")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])  # Maximum value of action possible in the environment for Hopper-v4 is 1
agent = TD3Agent(state_dim, action_dim, max_action)

num_episodes = 5000
gamma = 0.99
tau = 0.3
capacity = 50000
batch_size = 256
replay_buffer = PrioritizedReplayBuffer(capacity=capacity)
total_steps = 0 
reward_history = []

 
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action = agent.select_action(state,total_steps=total_steps) # Select an action with input state
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_steps += 1

        replay_buffer.add((state, action, reward, next_state, done))

        if len(replay_buffer.buffer) >= batch_size:
            states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size)

            # Compute target Q with action noise
            with torch.no_grad():
                target_actions = agent.actor_target(next_states)
                q1_target, q2_target = agent.critic_target(next_states, target_actions)
                min_Q = torch.min(q1_target, q2_target)
                target_value = rewards + (gamma) * (1 - dones) * min_Q

            with torch.no_grad():
                priority_q1, priority_q2 = agent.critic(states, actions)
                td_errors1 = (target_value - priority_q1).abs()
                td_errors2 = (target_value - priority_q2).abs()
                td_errors = torch.max(td_errors1, td_errors2)
                a = replay_buffer.update_priorities(indices, td_errors)
            # Update Critic with importance weights
            # Compute priorities without affecting gradients
            if episode % 500 == 0:
                
                with torch.no_grad():
                    # 1. Check weight magnitudes vs noise in all NoisyLinear layers
                    for i, layer in enumerate(agent.actor.net):
                        if isinstance(layer, NoisyLinear):
                            #layer.decay_noise = 1 * 0.5
                            w = layer.weight_mu
                            noise = layer.weight_epsilon
                            print(f"Layer {i}: weights μ={w.mean().item():.3f}±{w.std().item():.3f} | "
                                f"Noise ratio: {noise.std().item()/w.std().item():.3f}")

                    # 2. Verify target network stability
                    diff = sum((p1 - p2).abs().sum() 
                            for p1, p2 in zip(agent.actor.parameters(),
                                            agent.actor_target.parameters()))
                    print(f"Actor-target diff: {diff.item():.3f}")
                    
                with torch.no_grad():
                    # Ensure all tensors are on CPU before numpy conversion
                    agent.actor.eval()  # Disable noise
                    net_actions = agent.actor(states).cpu().numpy()
                    actions_np = actions.detach().cpu().numpy() 
                    agent.actor.train()  # Restore training mode
                    
                    action_noise = np.std(actions_np - net_actions)  # True policy variation
                    
                    # Parameter noise (already on correct device)
                    param_noise = torch.std(agent.actor.net[0].weight_epsilon).item()
                    
                    print(f"Param noise: {param_noise:.3f}, Action noise: {action_noise:.3f}, The Other noise: {agent.noise:.3f}")

                # Gradient analysis
                grads = [p.grad.norm().item() if p.grad is not None else 0 
                        for p in agent.actor.parameters()]
                print(f"Actor grad norms: {np.mean(grads):.2e} ± {np.std(grads):.2e}")
                
                
                

            # Main forward pass with gradients
            q1, q2 = agent.critic(states, actions)
            # if episode % 600 == 0:
            #     print(f'q1_target: {q1_target[:5]},\nmainQ1 {q1[:5]}')
            #     print(f'q2_target: {q2_target[:5]}, \nmainQ2: {q2[:5]}')
            critic_loss = torch.mean((torch.square(q1 - target_value) + torch.square(q2 - target_value)) * weights)
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()  # Only backward pass
            agent.critic_optimizer.step()
            # agent.critic_scheduler.step()

            # Delayed Actor Update
            if total_steps % 2 == 0:
                actor_loss = -torch.mean(agent.critic(states, agent.actor(states))[0])
                agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                agent.actor_optimizer.step()
                # agent.actor_scheduler.step()

                # Update target networks
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
        #print(f"Current Noise Level: {noise:.2f}")
        

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
}, "td3__with_PER.pth")



# The slow moving target network chooses the actions
# 

