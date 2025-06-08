#initialize enviroment
#n actions
#randomised reward gaussian dist

#initial reward values for a

#choose to be greedy with 1-epsillon or explore a with epsillon
#bandit(a) --> reward

#for chosen action:
#new reward = last reward + 1/k(actual reward - last reward)

#n = 10
#actions = [1,...,n]
#env = {action_count: [],
#       distribution_mean: [],
#       reward: []}

#def initialize():
#   for range(n):
#       env[action_count][i] = 0
#       env[reward][i] = 0  
#       env[distribution_mean][i] = random.num(1,n) 

#let epsilon = 0.01
#let steps = 1000

#def bandit(a):
# return gaussian_dist(env[distribution_mean][a],1).sample  

#def max_a():
#   return max(env[reward]).index

#for range(steps):
#   actual_reward = 0
#   action = 1
#   a* = max_a()

#   if random.random() > epsilon:
#       bandit(a*)--> actual_reward
#       action = a*
#       env[action_count][a*] = env[action_count][a*] + 1
#
#   else:
#       a = random.num(0,n)
#       bandit(a)--> actual_reward
#       action = a
#       env[action_count][a] = env[action_count][a] + 1

#   env[reward][action] = env[reward][action] + (1 / env[action_count][action]) * (actual_reward - env[reward][action]) 

import random
import matplotlib.pyplot as plt
import numpy as np

class epsilon_greeedy_bandit:

    def __init__(self, n_actions = 10, epsilon = 0.01):
        self.n = n_actions 
        self.epsilon = epsilon
        self.action_count = [0] * n_actions
        self.reward_estimate = [0] * n_actions
        self.true_mean = [random.uniform(1, self.n) for _ in range(self.n)]

    def bandit(self, action):
        
        return random.gauss(self.true_mean[action], 1)

    def choose_action(self):

        if random.random() > self.epsilon:
           
            return self.reward_estimate.index(max(self.reward_estimate))
       
        else:
           
            return random.randint(0,self.n -1)

    def update_estimates(self, action, reward):
        self.action_count[action] += 1
        alpha = 1 / self.action_count[action]
        self.reward_estimate[action] += alpha*(reward - self.reward_estimate[action])

    def run(self, steps = 1000):
        
        step = []
        rewards = []
        
        for i in range(steps):
            
            current_action = self.choose_action()
            current_reward = self.bandit(action=current_action)
            self.update_estimates(action=current_action,reward=current_reward) 
            rewards.append(current_reward)
            step.append(i)
        
        return rewards, step

bandits10 = epsilon_greeedy_bandit(n_actions=10,epsilon=0.01)
rewards, step = bandits10.run(steps=1000)

fig, ax = plt.subplots()

ax.plot(step, rewards)
plt.show()