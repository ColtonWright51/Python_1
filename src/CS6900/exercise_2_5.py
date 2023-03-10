"""
Created by Colton Wright on 2/3/2023
Exercise 2.5 from Sutton & Barto's "Reinforcement Learning: An Introduction
Second Edition" 
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import modules.easy_plots
# plt.style.use('dark_background')


# Make a class so that we can just call the different methods twice, one with
# sample averages, and one with action-value method.
class Bandits:

    step = 0 # What time step we are on, this is shared by all instances of Bandits class
    run = 0 # What run are we on, this is shared by all instances of Bandits class

    def __init__(self, n_arms, n_steps, n_runs, epsilon, alpha, name):
        self.n_arms = n_arms
        self.n_steps = n_steps
        self.n_runs = n_runs
        self.epsilon = epsilon
        self.alpha = alpha
        self.name = name

        # These variables are for calculations etc. you don't need to put these in constructor
        self.Q = np.zeros((n_arms, n_steps))
        self.N = np.zeros((n_arms, n_steps))
        self.rewards_received = np.zeros(n_steps)
        self.actions_taken = np.zeros(n_steps)
        
        # These variables are going to save all the Q values across the number
        # of runs that we want. We are goind to shove Q into the page n_runs
        # once a run. Average all these pages together to get average Q and
        # N.
        self.Q_global = np.zeros((n_arms,n_steps,n_runs))
        self.N_global = np.zeros((n_arms,n_steps,n_runs))

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
        plt.title(self.name + "_reward")
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
        plt.title(self.name + "_average_reward")
        modules.easy_plots.save_fig(self.name + "_average_reward")
    
    def plot_Q_max(self):
        Q_max_arr = np.zeros(n_steps)
        for i in range(self.n_steps):
            Q_max_arr[i] = np.max(self.Q[:, i])
        plt.figure()
        plt.plot(Q_max_arr)
        plt.title(self.name + "_Q_max")
        modules.easy_plots.save_fig(self.name + "_Q_max")

    def get_Q_max(self):
        Q_max_arr = np.zeros(n_steps)
        for i in range(self.n_steps):
            Q_max_arr[i] = np.max(self.Q[:, i])
        return Q_max_arr

    # Plot average Q_max of all the runs!
    def plot_global_Q_max(self, fig):
        Q_global_mean = np.mean(self.Q_global, axis=2)
        Q_global_max = np.zeros((n_steps))

        for i in range(self.n_steps):
            Q_global_max[i] = np.max(Q_global_mean[:, i])
        plt.figure(fig)
        plt.plot(Q_global_max, label=self.name)
        plt.title(self.name + "_Q_global_max")

    def plot_global_Q_max_no_fig(self, ax):
        Q_global_mean = np.mean(self.Q_global, axis=2)
        Q_global_max = np.zeros((n_steps))
        for i in range(self.n_steps):
            Q_global_max[i] = np.max(Q_global_mean[:, i])
        ax.plot(Q_global_max, label=self.name)


    def get_global_Q_max(self):
        Q_global_mean = np.mean(self.Q_global, axis=2)
        Q_global_max = np.zeros((n_steps))

        for i in range(self.n_steps):
            Q_global_max[i] = np.max(Q_global_mean[:, i])
        return Q_global_max


# Unused for now
class Levers():
    def __init__(self, mu, sigma):
        self.mu = mu # Mean
        self.sigma = sigma # Std. Dev.


def plot_levers(mu_list, sigma_list, run_num = 0):
    num_samps = 100
    values = np.zeros((len(mu_list),num_samps))
    for i in range(len(mu_list)):
        distribution = np.random.normal(mu_list[i], sigma_list[i], size=num_samps)
        values[i] = distribution
    plt.figure()
    sns.violinplot(values.T,inner='stick')
    if run_num == 0: # Init lever
        plt.title("Levers at " + str(Bandits.step))
        modules.easy_plots.save_fig("levers_step_"+str(Bandits.step))
    else:

        plt.title("Levers at end of run " + str(run_num))
        modules.easy_plots.save_fig("levers_run_"+str(run_num))

def get_reward(mu_list, sigma_list):
    # Get an array with the rewards of each lever for this time step
    reward = np.zeros(len(mu_list))
    for i in range(len(mu_list)):
        reward[i] = np.random.normal(mu_list[i], sigma_list[i], 1)
    return reward

def plot_globals_bandits(list_of_bandits):
    # plt.figure()
    # for i in list_of_bandits:
    #     Q_global_max = i.get_global_Q_max()
    #     plt.plot(Q_global_max, label=i.name)
    # plt.legend()
    # plt.title("Bandits global Q max")
    # modules.easy_plots.save_fig("Q_global_max")

    plt.figure()
    for i in list_of_bandits:
        i.plot_global_Q_max(plt.gcf().number)
    plt.legend()
    plt.title("Bandits global Q max")
    modules.easy_plots.save_fig("Q_global_max")

    # for i in list_of_bandits:
    #     plt.plot(i.get_avg_reward())
    # plt.title("Bandits average reward")
    # modules.easy_plots.save_fig("bandits_avg_reward")

    # plt.figure()
    # for i in list_of_bandits:
    #     plt.plot(i.get_Q_max(), label=i.name)
    # plt.legend()
    # plt.title("Bandits Q max")
    # modules.easy_plots.save_fig("bandits_Q_max")

def plot_bandits_and_levers(list_of_bandits, lever_mu, lever_sigma):
    plt.figure()
    ax1 = plt.subplot(211)
    for i in list_of_bandits:
        i.plot_global_Q_max_no_fig(ax1)


    # ax2 = plt.subplot(212, sharex=ax1)
    # num_samps = 100
    # values = np.zeros((len(lever_mu),num_samps))
    # for i in range(len(lever_mu)):
    #     distribution = np.random.normal(lever_mu[i], lever_sigma[i], size=num_samps)
    #     values[i] = distribution
    # plt.figure()
    # sns.violinplot(values.T,inner='stick')
    # if Bandits.run == 0: # Init lever
    #     plt.title("Levers at " + str(Bandits.step))
    #     modules.easy_plots.save_fig("levers_step_"+str(Bandits.step))
    # else:

    #     plt.title("Levers at end of run " + str(Bandits.run))
    #     modules.easy_plots.save_fig("levers_run_"+str(Bandits.run))

    plt.legend()
    plt.title("Bandits for run " + str(Bandits.run))
    modules.easy_plots.save_fig("run_"+ str(Bandits.run))


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
n_runs = 5
epsilon1 = .1
epsilon2 = 0
alpha1 = 0
alpha2 = .1

# Init bandits
bandit1 = Bandits(n_arms, n_steps, n_runs, epsilon1, alpha1, "bandit1") # Agent 1
bandit2 = Bandits(n_arms, n_steps, n_runs, epsilon2, alpha2, "bandit2") # Agent 2

# Init levers
lower_mu, upper_mu = -1, 10
lower_sigma, upper_sigma = 1, 1
random_mu, random_sigma = 0, .01 # Random walks the lever takes each step

# This was for random value & random distribution levers. We want equal levers at the beginnning
# lever_mu = np.zeros(n_arms) + (np.random.random(n_arms)*(upper_mu-lower_mu)+lower_mu)
# lever_sigma = np.zeros(n_arms) + (np.random.random(n_arms)*(upper_sigma-lower_sigma)+lower_sigma)

lever_mu = np.zeros(n_arms) + 1
lever_sigma = np.zeros(n_arms) + .01

plot_levers(lever_mu, lever_sigma) # Visualize levers at start

for j in range(n_runs):
    i = 1
    print("j: " , j)

    for i in range(1, n_steps):
        reward_array = get_reward(lever_mu, lever_sigma) # Same reward this step

        # Sample average
        B1_A = bandit1.sample_average_choose_action()
        bandit1.actions_taken[i] = B1_A
        bandit1.rewards_received[i] = reward_array[B1_A]
        bandit1.N[:, i] = bandit1.N[:, i-1]
        bandit1.Q[:, i] = bandit1.Q[:, i-1]
        bandit1.N[B1_A][i] = bandit1.N[B1_A, i] + 1
        bandit1.Q[B1_A][i] = bandit1.Q[B1_A, i]+1/bandit1.N[B1_A, i]*(bandit1.rewards_received[i]-bandit1.Q[B1_A, i])


        B2_A = bandit2.sample_average_choose_action()
        bandit2.actions_taken[i] = B2_A
        bandit2.rewards_received[i] = reward_array[B2_A]
        bandit2.N[:, i] = bandit2.N[:, i-1]
        bandit2.Q[:, i] = bandit2.Q[:, i-1]
        bandit2.N[B2_A][i] = bandit2.N[B2_A][i] + 1

        # Calculate the new expected reward for bandit 2, constant step-size parameter method.
        # The n in eq (2.6) in Sutton is the number of times this action has been chosen. So you
        # are summing over # of pulls, not what step you are on.

        # This array is what we will sum up.
        to_sum = np.zeros(bandit2.N[B2_A, i]) # Terms to sum up == # of times we pulled this lever
        to_sum[0] = (1-bandit2.alpha)**(bandit2.N[B2_A, i])*bandit2.Q[B2_A][0] # First term always the same

        # Need to keep building to_sum




        # There should be alpha here soon
        bandit2.Q[B2_A][i] = 1




        bandit1.Q[B2_A][i] = bandit1.Q[B2_A][i]+1/bandit2.N[B2_A][i]*(bandit2.rewards_received[i]-bandit2.Q[B2_A][i])


        Bandits.step = Bandits.step + 1
        lever_mu = lever_mu + np.random.normal(random_mu, random_sigma, n_arms) # Walk levers




    Bandits.run = Bandits.run + 1
    bandit1.Q_global[:,:,j] = bandit1.Q[:,:] # Save that run
    bandit2.Q_global[:,:,j] = bandit2.Q[:,:]
    plot_bandits_and_levers([bandit1, bandit2], lever_mu, lever_sigma) # Visualize levers at end of each run
    # Reset steps
    Bandits.step = 0
    # Reset the levers for the next run...
    lever_mu = np.zeros(n_arms) + 1
    lever_sigma = np.zeros(n_arms) + .01

    # Reset Q and N for next runs...
    bandit1.Q = np.zeros((n_arms, n_steps))
    bandit1.N = np.zeros((n_arms, n_steps))

    bandit2.Q = np.zeros((n_arms, n_steps))
    bandit2.N = np.zeros((n_arms, n_steps))


plot_globals_bandits([bandit1, bandit2])
plt.show()

# if __name__ == '__main__':
#     main()
