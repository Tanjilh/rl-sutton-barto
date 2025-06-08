import numpy as np
import matplotlib.pyplot as plt

class banditAlgo:
    def __init__(self, n_actions = 10, stationary = True, walk_std = 0.01):
        self.n = n_actions
        self.walk_std = walk_std
        self.stationary = stationary
        self.true_value = np.zeros(n_actions)

    def random_walk(self):
        self.true_value += np.random.normal(loc=0, scale= self.walk_std, size=self.n)

    def get_reward(self, action):
        return np.random.normal(self.true_value[action], self.walk_std)

    def optimal_actions(self):
        return np.argmax(self.true_value)
        
class epsilonGreedyAgent:
    def __init__(self, epsilon = 0.1, n_actions = 10, stationary=True):
        self.epsilon = epsilon
        self.n = n_actions
        self.stationary = stationary
        self.reward_estimate = np.zeros(n_actions)
        self.action_count = np.zeros(n_actions)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n)
        return np.argmax(self.reward_estimate)
        

    def update_estimates(self, action, reward):
        self.action_count[action] += 1
        if self.stationary:
            alpha = 1 / self.action_count[action]
        else:
            alpha = 0.1
        self.reward_estimate[action] += alpha*(reward - self.reward_estimate[action])

def run_algo(n_actions = 10, epsilon = 0.1, stationary = True, walk_std = 0.01, runs = 200, steps = 10000):
    avg_rewards = np.zeros(steps)
    optimal_action_percentage = np.zeros(steps)

    for run in range(runs):
    #iterate over runs
        bandit = banditAlgo(n_actions, walk_std)
        agent = epsilonGreedyAgent(epsilon,n_actions,stationary)
        rewards = np.zeros(steps)
        optimal_actions = np.zeros(steps)
        for i in range(steps):
        #iterate over steps
            action_select = agent.select_action()
            #for each step choose an action
            current_reward = bandit.get_reward(action=action_select)
            #return reward
            agent.update_estimates(action=action_select,reward=current_reward)
            #update reward estimate for arm
            bandit.random_walk()
            #update true reward values

            #store rewards in array
            rewards[i] = current_reward

            #check if action was optimal for step
            if action_select == bandit.optimal_actions():
                optimal_actions[i] = 1

        avg_rewards += rewards
        optimal_action_percentage += optimal_actions

    avg_rewards /= runs
    optimal_action_percentage = (optimal_action_percentage/runs)*100
    return avg_rewards, optimal_action_percentage


#run experiments
avg_rewards_stationary, optimal_action_stationary = run_algo(n_actions=10, epsilon=0.1, stationary=True, walk_std= 0.01, runs=200, steps=10000)
avg_rewards_nonstationary, optimal_action_nonStationary = run_algo(n_actions=10, epsilon=0.1, stationary=False, walk_std= 0.01, runs=200, steps=10000)

#plot
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(avg_rewards_stationary, label='Sample Average')
plt.plot(avg_rewards_nonstationary, label='Constant Step Size α=0.1')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Average Reward Over Time')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(optimal_action_stationary, label='Sample Average')
plt.plot(optimal_action_nonStationary, label='Constant Step Size α=0.1')
plt.xlabel('Steps')
plt.ylabel('% Optimal Action')
plt.title('Optimal Action Over Time')
plt.legend()
plt.grid(True)

plt.suptitle('ε-Greedy on Nonstationary Bandit Problem (ε = 0.1)', fontsize=14)
plt.tight_layout()
plt.show()