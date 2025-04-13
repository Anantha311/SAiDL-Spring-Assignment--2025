
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import torch.nn.functional as F
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NoisyLinear(nn.Module): # Making a Linear neural network which has noise
    def __init__(self, in_features, out_features, std_init=0.1,initial_decay_noise = 1,noise_dist='gamma', noise_multiplier = 0.2 ):
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
        self.noise_dist = noise_dist
        self.noise_multiplier = noise_multiplier

        self.reset_parameters()
        self.reset_noise()
        
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight_mu, gain=0.1)  # Smaller initialization
        nn.init.uniform_(self.bias_mu, -0.01, 0.01)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))

    def reset_noise(self):
        # #Truncated normal noise with controlled variance
        if self.noise_dist == 'gaussian':
            weight_noise = torch.randn_like(self.weight_epsilon) * self.noise_multiplier
            bias_noise = torch.randn_like(self.bias_epsilon) * self.noise_multiplier

        elif self.noise_dist == 'beta':
            # Beta distribution (symmetric version)
            beta_dist = torch.distributions.Beta(
                torch.tensor([2.0], device=self.weight_epsilon.device),
                torch.tensor([2.0], device=self.weight_epsilon.device)
            )
            weight_noise = (beta_dist.sample(self.weight_epsilon.shape).squeeze() - 0.5 ) * self.noise_multiplier
            bias_noise = (beta_dist.sample(self.bias_epsilon.shape).squeeze() - 0.5 ) * self.noise_multiplier

        elif self.noise_dist == 'gamma':
            # Centered gamma distribution
            device = self.weight_epsilon.device
            g1 = torch.distributions.Gamma(
                torch.tensor([2.0], device=device),
                torch.tensor([1.0], device=device)
            ).sample((self.out_features, self.in_features)).squeeze()
            g2 = torch.distributions.Gamma(
                torch.tensor([2.0], device=device),
                torch.tensor([1.0], device=device)
            ).sample((self.out_features, self.in_features)).squeeze()
            weight_noise = ((g1 - g2) / 2.0) * self.noise_multiplier
            bias_noise = ((g1.mean(dim=1) - g2.mean(dim=1)) / 2.0 ) * self.noise_multiplier # For bias [out_features]

        elif self.noise_dist == 'uniform':
            # Uniform distribution between a specified range
            weight_noise = (torch.empty_like(self.weight_epsilon).uniform_(-1.0, 1.0)) * self.noise_multiplier
            bias_noise = (torch.empty_like(self.bias_epsilon).uniform_(-0.5, 0.5)) * self.noise_multiplier

        elif self.noise_dist == 'laplace':
            # Laplace distribution (double exponential)
            laplace = torch.distributions.Laplace(
                torch.tensor([0.0], device=self.weight_epsilon.device),
                torch.tensor([1.0], device=self.weight_epsilon.device)
            )
            weight_noise = (laplace.sample(self.weight_epsilon.shape).squeeze() ) * self.noise_multiplier
            bias_noise = (laplace.sample(self.bias_epsilon.shape).squeeze()) * self.noise_multiplier

        elif self.noise_dist == 'exponential':
            # Exponential distribution (shifted to center around zero)
            exponential = torch.distributions.Exponential(
                torch.tensor([1.0], device=self.weight_epsilon.device)
            )
            # Subtracting the mean (1.0) centers the noise around zero.
            weight_noise = (exponential.sample(self.weight_epsilon.shape).squeeze() - 1.0) * self.noise_multiplier
            bias_noise = (exponential.sample(self.bias_epsilon.shape).squeeze() - 1.0) * self.noise_multiplier

        elif self.noise_dist == 'cauchy':
            # Cauchy distribution, which has heavy tails
            cauchy = torch.distributions.Cauchy(
                torch.tensor([0.0], device=self.weight_epsilon.device),
                torch.tensor([1.0], device=self.weight_epsilon.device)
            )
            weight_noise = (cauchy.sample(self.weight_epsilon.shape).squeeze()) * self.noise_multiplier
            bias_noise = (cauchy.sample(self.bias_epsilon.shape).squeeze()) * self.noise_multiplier

        elif self.noise_dist == 'student_t':
            # Student's T distribution, useful when expecting heavier tails than Gaussian
            # The degrees of freedom (df) parameter controls the tail weight.
            student_t = torch.distributions.StudentT(
                torch.tensor([3.0], device=self.weight_epsilon.device)
            )
            weight_noise = (student_t.sample(self.weight_epsilon.shape).squeeze()) * self.noise_multiplier
            bias_noise = (student_t.sample(self.bias_epsilon.shape).squeeze()) * self.noise_multiplier

        # Clamp the generated noise to keep it within desired bounds.
        self.bias_epsilon.data = bias_noise.clamp(-1.0, 1.0)
        self.weight_epsilon.data = weight_noise.clamp(-1.0, 1.0)


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
    def __init__(self, state_dim, action_dim, max_action,noise_dist='gaussian'): 
        super().__init__()
        self.net = nn.Sequential( # Neural Network of actor, It is a neural netwrk which contains weights such that your input will of states will give give outputs of action best to get the best reward
            NoisyLinear(state_dim, 256,noise_dist=noise_dist),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.LayerNorm(256),
            NoisyLinear(256, 256,noise_dist=noise_dist),
            nn.ReLU(),
            NoisyLinear(256, action_dim,noise_dist=noise_dist),
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
    def __init__(self, state_dim, action_dim, max_action, noise_type='gaussian'):

        self.actor = Actor(state_dim, action_dim, max_action,noise_dist=noise_type).to(device) # Main Actor network of TD3 Agent
        self.critic = Critic(state_dim, action_dim).to(device) # Main critic network of TD3 Agent

        # Target Networks
        self.actor_target = Actor(state_dim, action_dim, max_action,noise_dist=noise_type).to(device) # Target Actor network of TD3 Agent
        self.critic_target = Critic(state_dim, action_dim).to(device) # Target Critic network of TD3 Agent

        # Optimizers
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=3e-5) # Optimizer of actor
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=3e-5)# Optimizer of critic
        # Copy weights to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        #These lines copy the weights from the main networks (actor & critic) to their respective target networks (actor_target & critic_target). This is done initially so that it is same as main network
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


class ReplayBuffer:
    def __init__(self, capacity, n_step=3, gamma=0.99,device='cuda'):
        self.capacity = capacity # capacity: Maximum number of experiences the buffer can store.
        self.buffer = [] # buffer: Stores experiences (state, action, reward, next state, done).
        self.priorities = torch.zeros(capacity, dtype=torch.float32, device=device)  # Store priorities of experiences (important for sampling).
        self.pos = 0
        self.device = device
        self.n_step =n_step
        self.n_step_buffer = deque(maxlen=n_step) # because of dequeIf a new transition is added and the buffer is full, the oldest transition is automatically removed. 
        self.gamma = gamma

    def add(self, transition): # This doesnt calculate for last 2 states before it ends
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

       
        if len(self.buffer) < self.capacity:
            self.buffer.append(full_transition)
        else:
            self.buffer[self.pos] = full_transition
            #If buffer is not full, add new experience.
            #If full, overwrite the oldest experience(Oldest Experience at 0 which is self.pos).

        # Adds the priority at self.pos
        self.pos = (self.pos + 1) % self.capacity  # Circular buffer
        # Goes on add self.pos till it reaches capacity then comes as 0 and adds then at 1 it adds (keep on replacing old experiences)


    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.FloatTensor(np.array(actions)).to(device),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)
        )

env = gym.make("Hopper-v4")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])  # Maximum value of action possible in the environment for Hopper-v4 is 1
num_episodes = 5000
reward_history_gaussian = []
reward_history_beta = []
reward_history_gamma= []
reward_history_uniform = []
reward_history_laplace = []
reward_history_exponential =  []
reward_history_cauchy =  []
reward_history_student_t =  []
reward_history_gaussian_avg = []
reward_history_beta_avg = []
reward_history_gamma_avg= []
reward_history_uniform_avg = []
reward_history_laplace_avg = []
reward_history_exponential_avg =  []
reward_history_cauchy_avg =  []
reward_history_student_t_avg =  []

noise_configs = [
    {'name': 'Gaussian', 'params': {'noise_dist': 'gaussian'}},
    {'name': 'Beta(2,2)', 'params': {'noise_dist': 'beta'}},
    {'name': 'Gamma(2,1)', 'params': {'noise_dist': 'gamma'}},
    {'name': 'Uniform', 'params': {'noise_dist': 'uniform'}},
    {'name': 'Laplace(0,1)', 'params': {'noise_dist': 'laplace'}},
    {'name': 'Exponential(1)', 'params': {'noise_dist': 'exponential'}},
    {'name': 'Cauchy(0,1)', 'params': {'noise_dist': 'cauchy'}},
    {'name': 'Student_t', 'params': {'noise_dist': 'student_t'}}
    
]

avg_no = 50
for config in noise_configs:
    metrics = {
        'param_noise_ratio': [],
        'action_noise_std': []
    }
    print(f"\nRunning experiment: {config['name']}")
    agent = TD3Agent(state_dim, action_dim, max_action,noise_type=config['params']['noise_dist'])
    total_steps = 0 
    gamma = 0.99
    tau = 0.3
    capacity = 50000
    batch_size = 256
    replay_buffer = ReplayBuffer(capacity=capacity)
    for episode in range(1,num_episodes + 1):
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
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                # Compute target Q with action noise
                with torch.no_grad():
                    target_actions = agent.actor_target(next_states)
                    q1_target, q2_target = agent.critic_target(next_states, target_actions)
                    min_Q = torch.min(q1_target, q2_target)
                    target_value = rewards + (gamma) * (1 - dones) * min_Q

                if episode % 50 == 0:                       
                    with torch.no_grad():
                        # Ensure all tensors are on CPU before numpy conversion
                        agent.actor.eval()  # Disable noise
                        net_actions = agent.actor(states).cpu().numpy()
                        actions_np = actions.detach().cpu().numpy() 
                        agent.actor.train()  # Restore training mode
                        
                        action_noise = np.std(actions_np - net_actions)  # True policy variation
                        
                        # Parameter noise (already on correct device)
                        param_noise = torch.std(agent.actor.net[0].weight_epsilon).item()
                                        
                        metrics['param_noise_ratio'].append(param_noise)
                        metrics['action_noise_std'].append(action_noise)                    
                # Main forward pass with gradients
                q1, q2 = agent.critic(states, actions)

                critic_loss = F.mse_loss(q1,target_value) + F.mse_loss(q2,target_value)
                agent.critic_optimizer.zero_grad()
                critic_loss.backward()  # Only backward pass
                agent.critic_optimizer.step()

                # Delayed Actor Update
                if total_steps % 2 == 0:
                    actor_loss = -torch.mean(agent.critic(states, agent.actor(states))[0])
                    agent.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    agent.actor_optimizer.step()

                    # Update target networks
                    with torch.no_grad():
                        for param, target_param in zip(agent.critic.parameters(), agent.critic_target.parameters()):
                            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                        for param, target_param in zip(agent.actor.parameters(), agent.actor_target.parameters()):
                            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            state = next_state
            episode_reward += reward
        

        if config['name'] == "Gaussian":
            reward_history_gaussian.append(episode_reward)
 # # np.mean(reward_history[-100:]) calculates the average (mean) of the last 100 elements in the reward_history list.
        elif config['name'] == "Beta(2,2)":
            reward_history_beta.append(episode_reward)     
        elif config['name'] == "Gamma(2,1)":
            reward_history_gamma.append(episode_reward)   
        elif config['name'] == "Uniform":
            reward_history_uniform.append(episode_reward)        
        elif config['name'] == "Laplace(0,1)":
            reward_history_laplace.append(episode_reward)   
        elif config['name'] == "Exponential(1)":
            reward_history_exponential.append(episode_reward)     
        elif config['name'] == "Cauchy(0,1)":
            reward_history_cauchy.append(episode_reward)   
        elif config['name'] == "Student_t":
            reward_history_student_t.append(episode_reward)     


        if episode % avg_no  == 0:
            if config['name'] == "Gaussian":
                avg_reward = np.mean(reward_history_gaussian[-avg_no:])
                reward_history_gaussian_avg.append(avg_reward )# np.mean(reward_history[-100:]) calculates the average (mean) of the last 100 elements in the reward_history list.
            elif config['name'] == "Beta(2,2)":
                avg_reward = np.mean(reward_history_beta[-avg_no:])
                reward_history_beta_avg.append(np.mean(avg_reward))
            elif config['name'] == "Gamma(2,1)":
                avg_reward = np.mean(reward_history_gamma[-avg_no:])
                reward_history_gamma_avg.append(avg_reward)   
            elif config['name'] == "Uniform":
                avg_reward = np.mean(reward_history_uniform[-avg_no:])
                reward_history_uniform_avg.append(avg_reward)  
            elif config['name'] == "Laplace(0,1)":
                avg_reward = np.mean(reward_history_laplace[-avg_no:])
                reward_history_laplace_avg.append(avg_reward)   
            elif config['name'] == "Exponential(1)":
                avg_reward = np.mean(reward_history_exponential[-avg_no:])
                reward_history_exponential_avg.append(avg_reward) 
            elif config['name'] == "Cauchy(0,1)":
                avg_reward = np.mean(reward_history_cauchy[-avg_no:])
                reward_history_cauchy_avg.append(avg_reward)   
            elif config['name'] == "Student_t":
                avg_reward = np.mean(reward_history_student_t[-avg_no:])
                reward_history_student_t_avg.append(avg_reward)     

            print(f"Episode {episode}, Avg Reward: {avg_reward}, total Steps:{total_steps}")
            #print(f"Current Noise Level: {noise:.2f}")
    metrics_path = f"weights/td3_main_{config['name']}" 
    os.makedirs(metrics_path ,exist_ok=True) 
    fig_path =   f"weights/td3_main"    
    model_path = f"weights/td3_main_{config['name']}/td3_main_{config['name']}.pth"

    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
        'actor_target': agent.actor_target.state_dict(),
        'critic_target': agent.critic_target.state_dict(),
        'actor_optimizer': agent.actor_optimizer.state_dict(),
        'critic_optimizer': agent.critic_optimizer.state_dict(),
    }, model_path )
    with open(os.path.join(metrics_path, "metrics.txt"), "w") as f:
            f.write(f"Parameter Noise Ratio: {np.mean(metrics['param_noise_ratio']):.4f}\n")
            f.write(f"Global Noise Avg: {np.mean(metrics['action_noise_std']):.4f}\n")
            f.write(f"Final 100-Step Noise Avg: {np.mean(metrics['action_noise_std'][-100:]):.4f}\n")


# Plot learning progress
x_range = [((i  * avg_no)) for i in range(len(reward_history_gaussian_avg))]
plt.plot(x_range,reward_history_gaussian_avg,label="Gaussian")
plt.plot(x_range,reward_history_beta_avg,label = "Beta(2,2)")
plt.plot(x_range,reward_history_gamma_avg,label="Gamma(2,1)")
plt.plot(x_range,reward_history_uniform_avg,label="Uniform")
plt.plot(x_range,reward_history_laplace_avg,label = "Laplace(0,1)")
plt.plot(x_range,reward_history_exponential_avg,label="Exponential(1)")
plt.plot(x_range,reward_history_cauchy_avg,label="Cauchy(0,1)")
plt.plot(x_range,reward_history_student_t_avg,label = "Student_t")
plt.legend() 
plt.title(f"TD3 Learning Progress")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.savefig(fig_path)



