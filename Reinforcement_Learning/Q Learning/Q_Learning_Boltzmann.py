import gymnasium as gym
import numpy as np

# Discretization for observation spaces
def discretize_state(state, state_bins):
    return tuple(int(np.digitize(state[i], state_bins[i]) - 1) for i in range(len(state)))

def custom_reset(env):
    options = {
        "init_state": np.array([
            np.random.uniform(-2.0, 2.0),    
            np.random.uniform(-2.0, 2.0),    
            np.random.uniform(-0.05, 0.05),  
            np.random.uniform(-2.0, 2.0)     
        ])
    }
    state, _ = env.reset(options=options)
    return state, {}


def boltzmann_action(Q_values, tau): # Q_values.shape == (num_actions,)
    Q_values = np.clip(Q_values, -500, 500) 
    exp_values = np.exp(Q_values / tau)   
    probabilities = exp_values / np.sum(exp_values)  # Normalize
    if np.any(np.isnan(probabilities)):
        print(f"NaN detected in probabilities: {probabilities}")
        print(Q_values)
        probabilities = np.ones_like(probabilities) / len(probabilities)  # Fallback to uniform distribution
    return np.random.choice(len(Q_values), p=probabilities) # With that probablitites it chooses the action

def train_q_learning(env, state_bins, Q, alpha=0.1, gamma=0.99, tau=1.0, episodes=500):
    highest_reward = 0
    tau_decay = 0.995  # Decay tau gradually for less randomness over time
    min_tau = 2      # Minimum temperature for stability, this ideally should be between 0.1 and 0.5 say for good exploitation in the later stages but if i do that then the exp_values in boltzmann function becomes too big as Q = say [68,71] gets divided by 0.1 and becomes [680,710] and Nan is resulted as e^680 or e^710 is too big so keeping it 2 for now so that it isnt that high

    for episode in range(episodes):
        state, _ = custom_reset(env)
        state = discretize_state(state, state_bins)
        total_reward = 0

        for _ in range(500):
            action = boltzmann_action(Q[state], tau) 
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(next_state, state_bins)
            
            if not (terminated or truncated):
                Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
            
            state = next_state
            total_reward += reward

            if terminated or truncated:
                break
        
        highest_reward = max(highest_reward, total_reward)
        

        tau = max(min_tau, tau * tau_decay)

        #print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    print(f"Highest Reward Achieved: {highest_reward}")
    return Q

def test_q_learning(env, Q, state_bins, test_episodes=10):
    total_rewards = []

    for episode in range(test_episodes):
        state, _ = env.reset()
        state = discretize_state(state, state_bins)
        total_reward = 0

        for _ in range(200):  
            action = np.argmax(Q[state])  # Exploit only during testing
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(next_state, state_bins)
            
            total_reward += reward
            state = next_state
            
            env.render()

            if terminated or truncated:
                break
        
        total_rewards.append(total_reward)
        print(f"Test Episode {episode + 1}: Total Reward = {total_reward}")

    print(f"\nAverage Reward over {test_episodes} episodes: {np.mean(total_rewards):.2f}")

if __name__=="__main__":

    train_env = gym.make('CartPole-v1')  
    test_env = gym.make('CartPole-v1', render_mode='human')  

    number_state_bins = 20

    state_bins = [np.linspace(-4.8, 4.8, number_state_bins),
                  np.linspace(-4, 4, number_state_bins),
                  np.linspace(-0.418, 0.418, number_state_bins),
                  np.linspace(-4, 4, number_state_bins)]

    Q = np.random.uniform(low=0.5, high=1.0, size=[len(bins) for bins in state_bins] + [train_env.action_space.n])

    alpha = 0.2
    gamma = 0.99     
    tau = 10          # Initial Temperature for Exploration -> High to get a good exploration
    episodes = 2000   

    trained_Q = train_q_learning(train_env, state_bins, Q, alpha, gamma, tau, episodes)
    test_q_learning(test_env, trained_Q, state_bins)
    
    train_env.close()
    test_env.close()
