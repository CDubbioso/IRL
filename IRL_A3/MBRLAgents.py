#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning policies
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
from queue import PriorityQueue
from MBRLEnvironment import WindyGridworld

class DynaAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.Q_sa = np.zeros((n_states, n_actions))
        self.n_sa_s = np.zeros((n_states, n_actions, n_states))
        self.Rsum_sa_s = np.zeros((n_states, n_actions, n_states))

        
    def select_action(self, s, epsilon):
        if np.random.random() < epsilon:
            a = np.random.randint(0, self.n_actions)
        else:
            a = np.argmax(self.Q_sa[s])
        return a

    def update(self,s,a,r,done,s_next,n_planning_updates):
        # ---- Update model (Algorithm 2) ----
        self.n_sa_s[s, a, s_next] += 1
        self.Rsum_sa_s[s, a, s_next] += r

        # ---- Real-experience Q-learning update ----
        td_target = r + self.gamma * np.max(self.Q_sa[s_next])
        self.Q_sa[s, a] += self.learning_rate * (td_target - self.Q_sa[s, a])

        # ---- Planning loop ----
        if n_planning_updates <= 0:
            return

        # All (s,a) pairs that have been visited at least once: n(s,a) > 0
        n_sa = self.n_sa_s.sum(axis=2)
        observed = np.argwhere(n_sa > 0)
        if len(observed) == 0:
            return

        for _ in range(n_planning_updates):
            # Pick a random previously observed (s,a)
            idx = np.random.randint(len(observed))
            sp, ap = observed[idx]

            # Sample s' from estimated transition function p_hat(s'|s,a)
            counts = self.n_sa_s[sp, ap]
            total = counts.sum()
            probs = counts / total
            s_prime = np.random.choice(self.n_states, p=probs)

            # Estimated reward r_hat(s,a,s')
            r_hat = self.Rsum_sa_s[sp, ap, s_prime] / counts[s_prime]

            # Q-learning update with simulated transition
            td_target = r_hat + self.gamma * np.max(self.Q_sa[s_prime])
            self.Q_sa[sp, ap] += self.learning_rate * (td_target - self.Q_sa[sp, ap])

    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s]) # greedy action selection
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return

class PrioritizedSweepingAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, priority_cutoff=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.priority_cutoff = priority_cutoff
        
        self.queue = PriorityQueue()
        self.Q_sa = np.zeros((n_states, n_actions))
        self.n_sa_s = np.zeros((n_states, n_actions, n_states))
        self.Rsum_sa_s = np.zeros((n_states, n_actions, n_states))

    def select_action(self, s, epsilon):
        if np.random.random() < epsilon:
            a = np.random.randint(0, self.n_actions)
        else:
            a = np.argmax(self.Q_sa[s])
        return a

    def update(self,s,a,r,done,s_next,n_planning_updates):

        # ---- Update model (Algorithm 2) ----
        self.n_sa_s[s, a, s_next] += 1
        self.Rsum_sa_s[s, a, s_next] += r

        # ---- Compute priority for the real transition and possibly enqueue ----
        p = abs(r + self.gamma * np.max(self.Q_sa[s_next]) - self.Q_sa[s, a])
        if p > self.priority_cutoff:
            # PriorityQueue pops the smallest first, so we negate the priority
            self.queue.put((-p, (s, a)))

        # ---- Planning loop ----
        for _ in range(n_planning_updates):
            if self.queue.empty():
                break
            _, (sp, ap) = self.queue.get()

            counts = self.n_sa_s[sp, ap]
            total = counts.sum()
            if total == 0:
                continue

            # Sample s' from estimated transition function
            probs = counts / total
            s_prime = np.random.choice(self.n_states, p=probs)
            r_hat = self.Rsum_sa_s[sp, ap, s_prime] / counts[s_prime]

            # Q-learning update with simulated transition
            td_target = r_hat + self.gamma * np.max(self.Q_sa[s_prime])
            self.Q_sa[sp, ap] += self.learning_rate * (td_target - self.Q_sa[sp, ap])

            # ---- Loop over all predecessors (s_bar, a_bar) with n(s_bar, a_bar, sp) > 0 ----
            predecessors = np.argwhere(self.n_sa_s[:, :, sp] > 0)
            for s_bar, a_bar in predecessors:
                r_bar = self.Rsum_sa_s[s_bar, a_bar, sp] / self.n_sa_s[s_bar, a_bar, sp]
                p_bar = abs(r_bar + self.gamma * np.max(self.Q_sa[sp]) - self.Q_sa[s_bar, a_bar])
                if p_bar > self.priority_cutoff:
                    self.queue.put((-p_bar, (int(s_bar), int(a_bar))))

    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s]) # greedy action selection
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return        

def test():

    n_timesteps = 10001
    gamma = 1.0

    # Algorithm parameters
    policy = 'dyna' # or 'ps' 
    epsilon = 0.1
    learning_rate = 0.2
    n_planning_updates = 3

    # Plotting parameters
    plot = True
    plot_optimal_policy = True
    step_pause = 0.0001
    
    # Initialize environment and policy
    env = WindyGridworld()
    if policy == 'dyna':
        pi = DynaAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize Dyna policy
    elif policy == 'ps':    
        pi = PrioritizedSweepingAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize PS policy
    else:
        raise KeyError('Policy {} not implemented'.format(policy))
    
    # Prepare for running
    s = env.reset()  
    continuous_mode = False
    
    for t in range(n_timesteps):            
        # Select action, transition, update policy
        a = pi.select_action(s,epsilon)
        s_next,r,done = env.step(a)
        pi.update(s=s,a=a,r=r,done=done,s_next=s_next,n_planning_updates=n_planning_updates)
        
        # Render environment
        if plot:
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=plot_optimal_policy,
                       step_pause=step_pause)
            
        # Ask user for manual or continuous execution
        if not continuous_mode:
            key_input = input("Press 'Enter' to execute next step, press 'c' to run full algorithm")
            continuous_mode = True if key_input == 'c' else False

        # Reset environment when terminated
        if done:
            s = env.reset()
        else:
            s = s_next
            
    
if __name__ == '__main__':
    test()
