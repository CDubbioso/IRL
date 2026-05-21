import random
import numpy as np


class MCAgent():
    def __init__(self, env, gamma=0.9):
        self.env = env
        num_states = env.observation_space.n
        num_actions = env.action_space.n    
        self.num_actions = num_actions
        self.Q = np.zeros((num_states, num_actions))

        self.epsilon = 0.1
        self.alpha = 0.1

    def select_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.Q[state])
    
    def train(self, num_training_steps: int = 100_000):
        train_step = 0

        done = True
        s = self.env.reset()
        states_actions_rewards = []
        while True: 
            while not done:
                train_step += 1
                a = self.select_action(s)
                s_next, reward, done, _, _ = self.env.steps(a)
                states_actions_rewards.append((s, a, reward))
                s = s_next
            
            for s, a, r in reversed(states_actions_rewards):
                G = r + self.gamma * G 
                self._update_Q(s, a, G) 

            s = self.env.reset()
            done = False
            states_actions_rewards = []
            if train_step >= num_training_steps:
                break

    def _update_Q(self, s, a, G):
        self.Q[s, a] += 0.1 * (G - self.Q[s, a])


if __name__ == "__main__":
    env = SupermarketEnv()
    agent = MCAgent(env)
    agent.train()

    s = env.reset()
    for i in range(100):

        
        pass
