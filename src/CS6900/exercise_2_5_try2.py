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
    
    def reset(self):
        """
        Reset the bandit variables for the next run. Important to run this between runs.
        """

        self.Q = np.zeros((n_arms, n_steps))
        self.N = np.zeros((n_arms, n_steps))
        self.rewards_received = np.zeros(n_steps)
        self.actions_taken = np.zeros(n_steps)
        Bandits.step = 0

    def get_Q_max(self):
        # Get the maximum expected reward at each step in a run. Useful because
        # we can plot it and see which algorithm has a larger EXPECTED reward
        Q_max_arr = np.zeros(self.n_steps)
        for i in range(self.n_steps):
            Q_max_arr[i] = np.max(self.Q[:, i])
        return Q_max_arr

    def get_global_Q_max(self):
        # Same as above, but first we average every run together and then get
        # the largest expected turn of each step of that
        Q_global_mean = np.mean(self.Q_global, axis=2)
        Q_global_max = np.zeros((self.n_steps))

        for i in range(self.n_steps):
            Q_global_max[i] = np.max(Q_global_mean[:, i])
        return Q_global_max

#-----------------------------------------------------------------------------



def get_reward(mu_list, sigma_list):
    # Get an array with the rewards of each lever for this time step
    reward = np.zeros(len(mu_list))
    for i in range(len(mu_list)):
        reward[i] = np.random.normal(mu_list[i], sigma_list[i], 1)
    return reward

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

def plot_bandits(list_of_bandits):
    
    # Plot expected reward of greedy action vs time
    f, (ax1,ax2) = plt.subplots(1,2, sharex=True)
    for i in list_of_bandits:
        this_qmax = i.get_Q_max()
        ax1.plot(this_qmax, label=i.name)


    ax1.set_title("Bandits max(Q) for run " + str(Bandits.run))

    # PLot chosen action vs time, make sure we are exploring
    for i in list_of_bandits:
        chosen_action = i.actions_taken
        ax2.plot(chosen_action, label=i.name)
    
    plt.legend()
    ax2.set_title("Bandits chosen action for run " + str(Bandits.run))
    modules.easy_plots.save_fig("run_"+ str(Bandits.run))


def plot_globals_bandits(list_of_bandits):

    plt.figure()
    for i in list_of_bandits:
        q_max = i.get_global_Q_max()
        plt.plot(q_max, label=i.name)
    plt.legend()
    plt.title("Bandits global Q max")
    modules.easy_plots.save_fig("Q_global_max")


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
n_steps = 200
n_runs = 5
epsilon1 = .1
epsilon2 = .1
alpha1 = 0
alpha2 = .1

# Init bandits
bandit1 = Bandits(n_arms, n_steps, n_runs, epsilon1, alpha1, "bandit1") # Agent 1
bandit2 = Bandits(n_arms, n_steps, n_runs, epsilon2, alpha2, "bandit2") # Agent 2

# Init levers
lower_mu, upper_mu = -1, 10
lower_sigma, upper_sigma = 1, 1
random_mu, random_sigma = 0, .01 # Random walks the lever takes each step

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
        bandit2.N[B2_A, i] = bandit2.N[B2_A, i] + 1
        # Calculate the new expected reward for bandit 2, constant step-size parameter method.
        # The n in eq (2.6) in Sutton is the number of times this action has been chosen. So you
        # are summing over # of pulls, not what step you are on.
        bandit2.Q[B2_A, i] = bandit2.Q[B2_A, i] + bandit2.alpha*(bandit2.rewards_received[i]-bandit2.Q[B2_A, i])

        Bandits.step = Bandits.step + 1
        lever_mu = lever_mu + np.random.normal(random_mu, random_sigma, n_arms) # Walk levers


    Bandits.run = Bandits.run + 1
    bandit1.Q_global[:,:,j] = bandit1.Q[:,:] # Save that run
    bandit2.Q_global[:,:,j] = bandit2.Q[:,:]
    plot_bandits([bandit1, bandit2]) # Visualize levers at end of each run
    # Reset bandits for the next run...
    bandit1.reset()
    bandit2.reset()
    # Reset the levers for the next run...
    lever_mu = np.zeros(n_arms) + 1
    lever_sigma = np.zeros(n_arms) + .01




plot_globals_bandits([bandit1, bandit2])
plt.show()

# if __name__ == '__main__':
#     main()
