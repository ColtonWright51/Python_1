"""
Created by Colton Wright on 2/3/2023
Exercise 2.5 from Sutton & Barto's "Reinforcement Learning: An Introduction
Second Edition" 
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Make a class so that we can just call the different methods twice, one with
# sample averages, and one with action-value method.
class Bandits():
    step = 0 # What time step we are on, this is shared by all instances of Bandits class
    def __init__(self, n_arms, n_steps, epsilon):
        self.n_arms = n_arms
        self.Q = np.zeros((n_arms, n_steps))
        self.N = np.ones((n_arms, n_steps))
        self.epsilon = epsilon
        self.step = 0

    def print_hello(self):
        print("Hello")

    def increment_step(self):
        self.step = self.step+1

    def sample_average_choose_action(self):
        random_float = np.random.random()
        if random_float >= (self.epsilon):
            action = np.argmax(self.Q[:, self.step])
        else: # Choose randomly among your actions
            action = np.random.randint(0, n_arms)
        return action

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
    sns.violinplot(values.T,inner='stick')
    # plt.show()

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
n_steps = 50
epsilon = .1

# Init variables
bandit1 = Bandits(n_arms, n_steps, epsilon) # Agent 1
bandit2 = Bandits(n_arms, n_steps, epsilon) # Agent 2

lower_mu, upper_mu = -1, 1
lever_mu = np.zeros(n_arms) + (np.random.random(n_arms)*(upper_mu-lower_mu)+lower_mu)
lower_sigma, upper_sigma = .1, .1
lever_sigma = np.zeros(n_arms) + (np.random.random(n_arms)*(upper_sigma-lower_sigma)+lower_sigma)
plot_levers(lever_mu, lever_sigma) # Visualize levers at start


for i in range(1, n_steps):
    reward_array = get_reward(lever_mu, lever_sigma) # Same reward this step
    B1_A = bandit1.sample_average_choose_action()
    B1_R = reward_array[B1_A]
    bandit1.N[:, i] = bandit1.N[:, i-1]
    bandit1.Q[:, i] = bandit1.Q[:, i-1]
    bandit1.N[B1_A][i] = bandit1.N[B1_A][i] + 1
    bandit1.Q[B1_A][i] = bandit1.Q[B1_A][i]+1/bandit1.N[B1_A][i]*(B1_R-bandit1.Q[B1_A][i])


    bandit1.increment_step()

plt.show()
# if __name__ == '__main__':
#     main()
