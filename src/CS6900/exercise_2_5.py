"""
Created by Colton Wright on 2/3/2023
Exercise 2.5 from Sutton & Barto's "Reinforcement Learning: An Introduction
Second Edition" 
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import modules.easy_plots

# Make a class so that we can just call the different methods twice, one with
# sample averages, and one with action-value method.
class Bandits:

    step = 0 # What time step we are on, this is shared by all instances of Bandits class

    def __init__(self, n_arms, n_steps, epsilon, name):
        self.n_arms = n_arms
        self.n_steps = n_steps
        self.Q = np.zeros((n_arms, n_steps))
        self.N = np.zeros((n_arms, n_steps))
        self.epsilon = epsilon
        self.rewards_received = np.zeros(n_steps)
        self.name = name

    def sample_average_choose_action(self):
        random_float = np.random.random()
        if random_float >= (self.epsilon):
            action = np.argmax(self.Q[:, self.step])
        else: # Choose randomly among your actions
            action = np.random.randint(0, n_arms)
        return action

    def plot_reward(self):
        plt.figure()
        plt.plot(self.rewards_received)
        modules.easy_plots.save_fig(self.name + "_reward")

    def get_avg_reward(self):
        window = 1000
        avg_reward = []
        for i in range(len(self.rewards_received-window+1)):
            avg_reward.append(np.mean(self.rewards_received[i:i+window]))
        return avg_reward
    
    def plot_avg_reward(self):
        avg_reward = self.get_avg_reward()
        plt.figure()
        plt.plot(avg_reward)
        modules.easy_plots.save_fig(self.name + "_average_reward")
    
    def plot_Q_max(self):
        Q_max_arr = np.zeros(n_steps)
        for i in range(self.n_steps):
            Q_max_arr[i] = np.max(self.Q[:, i])
        plt.figure()
        plt.plot(Q_max_arr)
        modules.easy_plots.save_fig(self.name + "_Q_max")



# Unused for now
class Levers():
    def __init__(self, mu, sigma):
        self.mu = mu # Mean
        self.sigma = sigma # Std. Dev.

def plot_levers(mu_list, sigma_list):
    num_samps = 100
    values = np.zeros((len(mu_list),num_samps))
    for i in range(len(mu_list)):
        distribution = np.random.normal(mu_list[i], sigma_list[i], size=num_samps)
        values[i] = distribution
    plt.figure()
    sns.violinplot(values.T,inner='stick')

def get_reward(mu_list, sigma_list):
    # Get an array with the rewards of each lever for this time step
    reward = np.zeros(len(mu_list))
    for i in range(len(mu_list)):
        reward[i] = np.random.normal(mu_list[i], sigma_list[i], 1)
    return reward

"""
We have two RL agents standing in front of several different levers. The
agents both pull levers each time step and use their algorithms to learn. The
levers will give the same rewards to both agents if they pull the same lever.
The levers start with equal reward and the reward takes small random walks
at every time step.
"""

# def main():

# CONSTANTS
n_arms = 10
n_steps = 10000
epsilon = .01

# Init variables
bandit1 = Bandits(n_arms, n_steps, epsilon, "bandit1") # Agent 1
bandit2 = Bandits(n_arms, n_steps, epsilon, "bandit2") # Agent 2

lower_mu, upper_mu = -1, 10
lever_mu = np.zeros(n_arms) + (np.random.random(n_arms)*(upper_mu-lower_mu)+lower_mu)
lower_sigma, upper_sigma = 1, 1
lever_sigma = np.zeros(n_arms) + (np.random.random(n_arms)*(upper_sigma-lower_sigma)+lower_sigma)
plot_levers(lever_mu, lever_sigma) # Visualize levers at start


for i in range(1, n_steps):
    reward_array = get_reward(lever_mu, lever_sigma) # Same reward this step
    B1_A = bandit1.sample_average_choose_action()
    bandit1.rewards_received[i] = reward_array[B1_A]
    bandit1.N[:, i] = bandit1.N[:, i-1]
    bandit1.Q[:, i] = bandit1.Q[:, i-1]
    bandit1.N[B1_A][i] = bandit1.N[B1_A][i] + 1
    bandit1.Q[B1_A][i] = bandit1.Q[B1_A][i]+1/bandit1.N[B1_A][i]*(bandit1.rewards_received[i]-bandit1.Q[B1_A][i])



    Bandits.step = Bandits.step + 1
bandit1.plot_reward()
bandit1.plot_avg_reward()
bandit1.plot_Q_max()
plt.show()
# if __name__ == '__main__':
#     main()
