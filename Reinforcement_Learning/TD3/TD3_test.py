
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
    def __init__(self, in_features, out_features, std_init=0.1,initial_decay_noise = 1,noise_dist='gamma' ):
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
            weight_noise = torch.randn_like(self.weight_epsilon)
            bias_noise = torch.randn_like(self.bias_epsilon) 

        elif self.noise_dist == 'beta':
            # Beta distribution (symmetric version)
            beta_dist = torch.distributions.Beta(
                torch.tensor([2.0], device=self.weight_epsilon.device),
                torch.tensor([2.0], device=self.weight_epsilon.device)
            )
            weight_noise = beta_dist.sample(self.weight_epsilon.shape).squeeze() - 0.5  
            bias_noise = beta_dist.sample(self.bias_epsilon.shape).squeeze() - 0.5 

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
            weight_noise = (g1 - g2) / 2.0
            bias_noise = (g1.mean(dim=1) - g2.mean(dim=1)) / 2.0  # For bias [out_features]

        elif self.noise_dist == 'uniform':
            # Uniform distribution between a specified range
            weight_noise = torch.empty_like(self.weight_epsilon).uniform_(-1.0, 1.0)
            bias_noise = torch.empty_like(self.bias_epsilon).uniform_(-0.5, 0.5)

        elif self.noise_dist == 'laplace':
            # Laplace distribution (double exponential)
            laplace = torch.distributions.Laplace(
                torch.tensor([0.0], device=self.weight_epsilon.device),
                torch.tensor([1.0], device=self.weight_epsilon.device)
            )
            weight_noise = laplace.sample(self.weight_epsilon.shape).squeeze()
            bias_noise = laplace.sample(self.bias_epsilon.shape).squeeze()

        elif self.noise_dist == 'exponential':
            # Exponential distribution (shifted to center around zero)
            exponential = torch.distributions.Exponential(
                torch.tensor([1.0], device=self.weight_epsilon.device)
            )
            # Subtracting the mean (1.0) centers the noise around zero.
            weight_noise = exponential.sample(self.weight_epsilon.shape).squeeze() - 1.0
            bias_noise = exponential.sample(self.bias_epsilon.shape).squeeze() - 1.0

        elif self.noise_dist == 'cauchy':
            # Cauchy distribution, which has heavy tails
            cauchy = torch.distributions.Cauchy(
                torch.tensor([0.0], device=self.weight_epsilon.device),
                torch.tensor([1.0], device=self.weight_epsilon.device)
            )
            weight_noise = cauchy.sample(self.weight_epsilon.shape).squeeze()
            bias_noise = cauchy.sample(self.bias_epsilon.shape).squeeze()

        elif self.noise_dist == 'student_t':
            # Student's T distribution, useful when expecting heavier tails than Gaussian
            # The degrees of freedom (df) parameter controls the tail weight.
            student_t = torch.distributions.StudentT(
                torch.tensor([3.0], device=self.weight_epsilon.device)
            )
            weight_noise = student_t.sample(self.weight_epsilon.shape).squeeze()
            bias_noise = student_t.sample(self.bias_epsilon.shape).squeeze()

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




# Load the trained actor network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

noise_configs = [
    {'name': 'Gaussian', 'params': {'noise_dist': 'gaussian'}},
    {'name': 'Beta(2,5)', 'params': {'noise_dist': 'beta'}},
    {'name': 'Gamma(2,1)', 'params': {'noise_dist': 'gamma'}},
    {'name': 'Uniform', 'params': {'noise_dist': 'uniform'}},
    {'name': 'Laplace(0,1)', 'params': {'noise_dist': 'laplace'}},
    {'name': 'Exponential(1)', 'params': {'noise_dist': 'exponential'}},
    {'name': 'Cauchy(0,1)', 'params': {'noise_dist': 'cauchy'}},
    {'name': 'Student_t', 'params': {'noise_dist': 'student_t'}}
    
]
num_episodes = 100 
seed = 42 

# Set global seeds for reproducibility
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

for config in noise_configs:
    print(f"Testing for {config['name']}")  
    
    # Create environment once per config
    env = gym.make("Hopper-v4", render_mode=None)  
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Seed environment
    env.reset(seed=seed)
    env.action_space.seed(seed)
    
    rewards = []
    
    try:
        # Model loading
        actor = Actor(state_dim, action_dim, max_action, noise_dist=config['params']['noise_dist']).to(device)
        checkpoint = torch.load(
            f"weights/td3_main_{config['name']}/td3_main_{config['name']}.pth",
            map_location=device
        )
        actor.load_state_dict(checkpoint["actor"])
        actor.eval()

        # Testing loop
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.FloatTensor(state).to(device)
                with torch.no_grad():
                    action = actor(state_tensor).cpu().numpy()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = next_state

            rewards.append(total_reward)
            if episode % 10 == 0:
                print(f"{config['name']} - Episode {episode+1}: Reward = {total_reward:.1f}")

        # Save metrics
        metrics_path = f"weights/td3_main_{config['name']}"
        os.makedirs(metrics_path, exist_ok=True)
        
        with open(os.path.join(metrics_path, "metrics.txt"), "a") as f:
            f.write(f"\nTest Results ({num_episodes} episodes):")
            f.write(f"\nAverage Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
            f.write(f"\nMax Reward: {np.max(rewards):.2f}")
            f.write(f"\nMin Reward: {np.min(rewards):.2f}\n")

    finally:
        env.close()