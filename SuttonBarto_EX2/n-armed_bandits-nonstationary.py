import random
import matplotlib.pyplot as plt
import numpy as np

class epsilon_greeedy_bandit:

    def __init__(self, n_actions = 10, epsilon = 0.1, stationary = True):
        self.n = n_actions 
        self.epsilon = epsilon
        self.stationary = stationary
        self.true_mean = [random.uniform(1, self.n) for _ in range(self.n)]

        if stationary:
            self.action_count = [0] * n_actions
            self.reward_estimate = [0] * n_actions
        else: 
            self.reward_estimate_nonStationary = [0] * n_actions
   
    def bandit(self, action):
        
        return random.gauss(self.true_mean[action], 1)

    def choose_action(self):

        if random.random() > self.epsilon:
           
           if self.stationary:
                return self.reward_estimate.index(max(self.reward_estimate))
           else:
               return self.reward_estimate_nonStationary.index(max(self.reward_estimate_nonStationary))
       
        else:
           
            return random.randint(0,self.n -1)

    def update_estimates(self, action, reward):

        if self.stationary:
            self.action_count[action] += 1
            alpha = 1 / self.action_count[action]
            self.reward_estimate[action] += alpha*(reward - self.reward_estimate[action])
        else:
            alpha = 0.1
            self.reward_estimate_nonStationary[action] += alpha*(reward - self.reward_estimate_nonStationary[action])

    def run(self, runs = 1 ,steps = 10000):
        
        step = []
        cumulative_reward = [0] * steps

        for run in range(runs):
            rewards = []

            for i in range(steps):  
                current_action = self.choose_action()
                current_reward = self.bandit(action=current_action)
                self.update_estimates(action=current_action,reward=current_reward) 
                rewards.append(current_reward)
                if run == 0:
                    step.append(i)      
                
                for j in range(self.n):
                    self.true_mean[j] += random.gauss(0,0.01)
            
            for k in range(steps):
                cumulative_reward[k] += rewards[k]

        for j in range(steps):
            cumulative_reward[j] /= runs
        
        return cumulative_reward, step

bandits_stationary = epsilon_greeedy_bandit(n_actions=10,epsilon=0.01,stationary=True)
rewards_stationary, step_stationary= bandits_stationary.run(steps=10000)

bandits_nonStationary = epsilon_greeedy_bandit(n_actions=10, epsilon=0.01,stationary=False)
rewards_nonstationary, step_nonstationary = bandits_nonStationary.run(steps=10000)

fig, ax = plt.subplots()

ax.plot(step_stationary, rewards_stationary, color = 'red')
ax.plot(step_nonstationary, rewards_nonstationary, color = 'blue')
plt.show()