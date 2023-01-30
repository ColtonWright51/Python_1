import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def bandit(action, mean_list, dev_list):
    reward = np.random.normal(mean_list[action], dev_list[action])
    return reward

def choose_action(Q, epsilon, num_arms):
    # print("choose_action")
    random_float = np.random.random()
    # print(random_float)
    if random_float >= (epsilon):
        action = np.argmax(Q)
    else: # Choose randomly among your actions
        action = np.random.randint(0,num_arms)
    return action

# def plot_distributions(mean_list, dev_list):

#     s = np.zeros(len(mean_list))
#     for i in range(mean_list):
#         s[i] = np.random.normal(mean_list, dev_list, 10)
#         print(s[i])
#     sns.catplot(data=df, x="age", y="class", kind="violin", color=".9", inner=None)
#     sns.swarmplot(data=df, x="age", y="class", size=3)

def main():

    num_arms = 5
    num_steps = 100000
    mean_list = np.array([0,1,-1,2,-.5])
    dev_list = np.array([1,2,1,1,1])

    Q = np.zeros(num_arms)
    # Q = np.array([0,1,2,3,4])
    N = np.ones(num_arms)
    epsilon = .5

    for i in range(num_steps):
        A = choose_action(Q, epsilon, num_arms)
        R = bandit(A, mean_list, dev_list)
        Q[A] = Q[A]+1/(N[A])*(R-Q[A])
        N[A] = N[A]+1
        print(Q)


def plot_testing():
    mean_list = np.array([0,1,-1,2,-.5])
    dev_list = np.array([1,2,1,1,1])
    # plot_distributions(mean_list, dev_list):

if __name__ == '__main__':
    main()