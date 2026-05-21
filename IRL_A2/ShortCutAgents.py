import numpy as np
from ShortCutEnvironment import ShortcutEnvironment, WindyShortcutEnvironment
# - - - - - - - - 

class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0, env: ShortcutEnvironment=None):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        # - - - - - - - 
        self.env: ShortcutEnvironment = env
        self.Q = np.zeros((n_states, n_actions))
        # - - - - - - - 
        
    def select_action(self, state):
        # TO DO: Implement policy
        action = None
        # - - - - - - - 
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            action = np.argmax(self.Q[state])
        # - - - - - - - 
        return action
        
    def update(self, state, action, reward, done, next_state): # Augment arguments if necessary
        # TO DO: Implement Q-learning update
        # - - - - - - - 
        if done:
            target = reward
        else: 
            target = reward + self.gamma * np.max(self.Q[next_state])
        
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
        return 
        # - - - - - - -         
    
    def train(self, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        # - - - - - - - 
        for _ in range(n_episodes):
            self.env.reset()
            cumulative_reward = 0 

            while not self.env.done(): 
                state = self.env.state()
                action = self.select_action(state)
                reward = self.env.step(action)
                next_state = self.env.state()
                cumulative_reward += reward
                
                self.update(state, action, reward, self.env.done(), next_state)
            
            episode_returns.append(cumulative_reward)
        # - - - - - - - 
        return episode_returns


class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0, env: ShortcutEnvironment=None):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        # - - - - - - - 
        self.env: ShortcutEnvironment = env
        self.Q = np.zeros((n_states, n_actions))
        # - - - - - - - 
        
    def select_action(self, state):
        # TO DO: Implement policy
        action = None
        # - - - - - - - 
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            action = np.argmax(self.Q[state])
        # - - - - - - - 
        return action
        
    def update(self, state, action, reward, done, next_state): # Augment arguments if necessary
        # TO DO: Implement SARSA update
        # - - - - - - - 
        if done:
            target = reward
            next_action = None
        else: 
            next_action = self.select_action(next_state)
            target = reward + self.gamma * self.Q[next_state, next_action]

        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
        return next_action
        # - - - - - - - 

    def train(self, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        # - - - - - - - 
        for _ in range(n_episodes):
            self.env.reset()
            cumulative_reward = 0 
            state = self.env.state()
            action = self.select_action(state)
            
            while not self.env.done():     
                reward = self.env.step(action)
                next_state = self.env.state()
                cumulative_reward += reward
                
                next_action = self.update(state, action, reward, self.env.done(), next_state)
                state, action = next_state, next_action
            
            episode_returns.append(cumulative_reward)
        # - - - - - - - 
        return episode_returns


class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        
    def select_action(self, state):
        # TO DO: Implement policy
        action = None
        return action
        
    def update(self, state, action, reward, done): # Augment arguments if necessary
        # TO DO: Implement Expected SARSA update
        pass

    def train(self, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        return episode_returns    


class nStepSARSAAgent(object):

    def __init__(self, n_actions, n_states, n, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.n = n
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        
    def select_action(self, state):
        # TO DO: Implement policy
        action = None
        return action
        
    def update(self, states, actions, rewards, done): # Augment arguments if necessary
        # TO DO: Implement n-step SARSA update
        pass
    
    def train(self, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        return episode_returns  
    
    
