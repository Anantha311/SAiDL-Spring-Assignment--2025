import gymnasium as gym
import numpy as np

# Discretization for observation spaces
def discretize_state(state, state_bins): # This function is to discretize teh space menaing the q learning can only work with instger tuples like (24,1,4,1) so it converts say decimal tuples like (0.23,0.12,0.95,0.76) to there respective integer according to where they lie in their respective line space
    return tuple(int(np.digitize(state[i], state_bins[i]) - 1) for i in range(len(state)))  # np.digitize() identifies which bin each element belongs to. The -1 is used to shift the bin indices to start from zero (since np.digitize() by default returns bin indices starting at 1).

def custom_reset(env): # To cusomly reset the environment to start from random state rather than the predefined state of reset 
    options = {
        "init_state": np.array([
            np.random.uniform(-2.0, 2.0),    # Cart position
            np.random.uniform(-2.0, 2.0),    # Cart velocity
            np.random.uniform(-0.05, 0.05),  # Pole angle
            np.random.uniform(-2.0, 2.0)     # Pole angular velocity
        ])
    }
    state, _ = env.reset(options=options)
    return state, {}

def train_q_learning(env, state_bins, Q, alpha=0.1, gamma=0.99, epsilon=1.0, episodes=500):
    highest_reward = 0
    epsilon_decay = 0.999 # to make X epsilon decrease to X*0.999 in every episode.  because of slow exploration decay it explores a lot in the first half and does action a lot in the second half
    min_epsilon = 0.05 # Not to decrease epsilon below this 

    for episode in range(episodes):
        state, _ = custom_reset(env)
        state = discretize_state(state, state_bins)
        total_reward = 0

        for _ in range(500):                      #This below generates a random number between 0 and 1
            action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q[state]) # if exploration else exploitation. In exploration it picks a random action  and in exploitation it chooses the action which has the better Q Value for the corresponding state
            
            next_state, reward, terminated, truncated, _ = env.step(action) 
            #This is what happens when above statement is run
            '''
            At each time step:
            The agent selects an action (e.g., move the cart left or right in CartPole).
            The environment updates based on that action.
            The environment returns:
            Observation (the new state).
            Reward (a score for that action).
            Termination flag (True if the episode ends).
            Truncation flag (True if the maximum steps are reached).
            '''
            next_state = discretize_state(next_state, state_bins)
            
            if not (terminated or truncated):
                Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action]) # Q-Learning Update Rule
            
            state = next_state
            total_reward += reward

            if terminated or truncated:
                break
        
        highest_reward = max(highest_reward, total_reward)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        #print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    print(f"Highest Reward Achieved: {highest_reward}")
    return Q

def test_q_learning(env, Q, state_bins, test_episodes=10): # The Q Value which is trained is now tested
    total_rewards = []

    for episode in range(test_episodes):
        state, _ = env.reset()
        state = discretize_state(state, state_bins)
        total_reward = 0

        for _ in range(200):  # Longer episodes for testing
            action = np.argmax(Q[state])  # Only exploit (greedy policy)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(next_state, state_bins)
            
            total_reward += reward
            state = next_state
            
            env.render()  # Render only during testing

            if terminated or truncated:
                break
        
        total_rewards.append(total_reward)
        print(f"Test Episode {episode + 1}: Total Reward = {total_reward}")

    print(f"\nAverage Reward over {test_episodes} episodes: {np.mean(total_rewards):.2f}")

if __name__=="__main__":

    # Initialize environments separately for training and testing
    train_env = gym.make('CartPole-v1')               # No rendering for training
    test_env = gym.make('CartPole-v1', render_mode='human')  # Rendering enabled only for testing

    number_state_bins = 20

    state_bins = [np.linspace(-4.8, 4.8, number_state_bins),
                  np.linspace(-4, 4, number_state_bins),
                  np.linspace(-0.418, 0.418, number_state_bins),
                  np.linspace(-4, 4, number_state_bins)]

    Q = np.random.uniform(low=0.5, high=1.0, size=[len(bins) for bins in state_bins] + [train_env.action_space.n])

    alpha = 0.2
    gamma = 0.99     
    epsilon = 0.1    
    episodes = 2000   

    trained_Q = train_q_learning(train_env, state_bins, Q, alpha, gamma, epsilon, episodes)
    test_q_learning(test_env, trained_Q, state_bins)
    
    train_env.close()
    test_env.close()


'''
Q-Table Initialization — Preventing Early Collapse
Before:
-> Initializing Q values near zero (low=0, high=0.01) makes all actions look equally bad.
-> The agent lacks confidence to explore or exploit effectively.
After:
-> Initializing Q values in a higher range (low=0.5, high=1.0) encourages exploration.
-> Higher Q-values create stronger initial incentives for the agent to try new actions.
'''