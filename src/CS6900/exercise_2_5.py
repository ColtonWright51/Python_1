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
    def __init__(self, n_arms, n_steps):
        self.Q = np.zeros(n_arms, n_steps)
        self.N = np.ones(n_arms, n_steps)








def main():
    print("HI")
    value = np.random.normal(loc=5,scale=.1,size=1000)
    x=5
    sns.violinplot(value)
    plt.show()
    print(dir())

if __name__ == '__main__':
    main()
