import numpy as np
import matplotlib.pyplot as plt




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


def main():

    num_arms = 5
    num_steps = 10000
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





if __name__ == '__main__':
    main()