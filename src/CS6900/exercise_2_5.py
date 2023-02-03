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
    x = [1,2,3]
    def __init__(self, n_arms, n_steps):
        self.Q = np.zeros((n_arms, n_steps))
        self.N = np.ones((n_arms, n_steps))

    def print_hello(self):
        print("Hello")






# def main():
bandit1 = Bandits(2, 5)
bandits2 = Bandits(3,3)
print(bandit1.x)
print(bandits2.x)
bandit1.x = [4,5,6]
print(bandit1.x)
print(bandits2.x)
# bandit1.print_hello()
value = np.random.normal(loc=5,scale=.1,size=100)
value2 = np.random.normal(loc=4,scale=.1,size=100)
value = [value, value2]
x=5
sns.violinplot(value, inner='stick')
plt.show()
print(dir())

# if __name__ == '__main__':
#     main()
