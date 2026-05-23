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

        # initialize elements for Dyna
        self.Q_sa = np.zeros((n_states, n_actions))
        self.n_sa_s = np.zeros((n_states, n_actions, n_states))
        self.Rsum_sa_s = np.zeros((n_states, n_actions, n_states))

        
    def select_action(self, s, epsilon):
        # e-greedy action selection
        if np.random.random() < epsilon:
            a = np.random.randint(0, self.n_actions)
        else:
            a = np.argmax(self.Q_sa[s])
        return a

    def update(self,s,a,r,done,s_next,n_planning_updates):
        self.n_sa_s[s, a, s_next] += 1      # update transition counts
        self.Rsum_sa_s[s, a, s_next] += r   # update reward sums

        # update Q-table 
        td_target = r + self.gamma * np.max(self.Q_sa[s_next])
        self.Q_sa[s, a] += self.learning_rate * (td_target - self.Q_sa[s, a])

        # planning loop 
        if n_planning_updates <= 0:
            return

        # (s,a) pairs that have been visited at least once: n(s,a) > 0
        n_sa = self.n_sa_s.sum(axis=2)
        observed = np.argwhere(n_sa > 0)
        if len(observed) == 0:
            return

        for _ in range(n_planning_updates):
            # pick a random previously observed (s,a)
            idx = np.random.randint(len(observed))
            sp, ap = observed[idx]

            # sample s' from estimated transition function p_hat(s'|s,a)
            counts = self.n_sa_s[sp, ap]
            total = counts.sum()
            probs = counts / total      # estimate transition function 
            s_prime = np.random.choice(self.n_states, p=probs)  

            # estimated reward function r_hat(s,a,s')
            r_hat = self.Rsum_sa_s[sp, ap, s_prime] / counts[s_prime]

            # update Q-table 
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
        
        # initialize elements for Prioritized Sweeping
        self.queue = PriorityQueue()
        self.Q_sa = np.zeros((n_states, n_actions))
        self.n_sa_s = np.zeros((n_states, n_actions, n_states))
        self.Rsum_sa_s = np.zeros((n_states, n_actions, n_states))

    def select_action(self, s, epsilon):
        # e-greedy action selection
        if np.random.random() < epsilon:
            a = np.random.randint(0, self.n_actions)
        else:
            a = np.argmax(self.Q_sa[s])
        return a

    def update(self,s,a,r,done,s_next,n_planning_updates):
        self.n_sa_s[s, a, s_next] += 1
        self.Rsum_sa_s[s, a, s_next] += r

        # compute priority 
        p = abs(r + self.gamma * np.max(self.Q_sa[s_next]) - self.Q_sa[s, a])
        # state-action needs update, add to queue
        if p > self.priority_cutoff:
            self.queue.put((-p, (s, a)))    # -p converts to a max-queue

        # planning loop 
        for _ in range(n_planning_updates):
            # sample PQ, break when empty
            if self.queue.empty():
                break
            _, (sp, ap) = self.queue.get()

            counts = self.n_sa_s[sp, ap]
            total = counts.sum()
            if total == 0:
                continue

            # sample s' from estimated transition function p_hat(s'|s,a)
            probs = counts / total
            # simulate model
            s_prime = np.random.choice(self.n_states, p=probs)
            r_hat = self.Rsum_sa_s[sp, ap, s_prime] / counts[s_prime]

            # update Q-table
            td_target = r_hat + self.gamma * np.max(self.Q_sa[s_prime])
            self.Q_sa[sp, ap] += self.learning_rate * (td_target - self.Q_sa[sp, ap])

            # predecessors loop (s_bar, a_bar) with n(s_bar, a_bar, sp) > 0 
            predecessors = np.argwhere(self.n_sa_s[:, :, sp] > 0)
            for s_bar, a_bar in predecessors:
                r_bar = self.Rsum_sa_s[s_bar, a_bar, sp] / self.n_sa_s[s_bar, a_bar, sp]            # get reward from model
                p_bar = abs(r_bar + self.gamma * np.max(self.Q_sa[sp]) - self.Q_sa[s_bar, a_bar])   # compute priority p 
                # state-action needs update, add to queue 
                if p_bar > self.priority_cutoff:
                    self.queue.put((-p_bar, (int(s_bar), int(a_bar))))  # -p_bar converts to a max-queue

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
