import gym
import numpy as np
import matplotlib.pyplot as plt
import math
import random


class QLearner:
    def ns(self, num, p):
        return int(round(self.env.observation_space.high[num] - self.env.observation_space.low[num] * p + 1))

    def __init__(self, learning, discount):
        # Import and initialize Mountain Car Environment
        self.env = gym.make('MountainCar-v0')
        self.env.reset()
        # Initialize learning rate (alpha) and discount value (gamma)
        self.learning = learning
        self.discount = discount
        # Down-scaling feature space to discrete range, determine number of discrete states
        self.num_states = []
        self.num_states.extend([self.ns(1, 100), self.ns(0, 10)])
        # TODO: YOUR CODE
        # Initiate Q-table with uniform random variables in [-1, 1]
        # print(self.num_states)
        self.Q = [[[random.uniform(-1, 1) for _ in range(self.env.action_space.n)] for _ in range(self.num_states[0])] for _ in range(self.num_states[1])]
        # TODO: YOUR CODE
        # print(self.Q)
        self.n = self.env.action_space.n
        # print(self.n)

    def dis_dif_state_ol(self, n, p,state):
        return int(round((state[n] - self.env.observation_space.low[n]) * p))
    # Get discrete representation of state
    def discretize_state(self, state):
        # TODO: YOUR CODE
        return [self.dis_dif_state_ol(0, 10,state), self.dis_dif_state_ol(1, 100,state)]

    # Determine next action using epsilon-greedy approach
    def get_action(self, state, epsilon):
        if random.random < epsilon:
            return random.randint(0, self.n - 1)
        else:
            return np.argmax(self.Q[state[0]][state[1]])


    # Adjust Q value for current state
    def update_q(self, state, action, new_state, reward):
        self.Q[state[0]][state[1]][action] = (1 - self.learning) * self.Q[state[0]][state[1]][action] + self.learning * (reward + self.discount * np.max(self.Q[new_state[0]][new_state[1]]))

    # Q-Learning function
    def q_learning(self, epsilon, min_eps, episodes):
        # Initialize variables to track rewards
        reward_list = []
        ave_reward_list = []
        epsilon_r = (epsilon - min_eps) / episodes

        # Run Q-learning algorithm
        for i in range(episodes):
            # Initialize parameters
            done = False
            tot_reward, reward = 0, 0
            state = self.env.reset()
            disc_state = self.discretize_state(state)

            while not done:
                # Render environment for last twenty episodes
                if i >= (episodes - 20):
                    self.env.render()

                # Determine next action, state, and reward
                # TODO: YOUR CODE
                action = self.get_action(disc_state, epsilon)
                new_state, reward, done, _ = self.env.step(action)
                disc_new_state = self.discretize_state(new_state)

                # Update Q
                if done and new_state[0] >= 0.5:
                    self.Q[disc_state[0]][disc_state[1]][action] = reward
                else:
                    self.update_q(disc_state, action, disc_new_state, reward)

                # Update variables
                # TODO: YOUR CODE
                disc_state = disc_new_state
                tot_reward = tot_reward + reward

            # Decay epsilon
            # TODO: YOUR CODE
            if min(min_eps, epsilon) == min_eps:
                epsilon -= epsilon_r
            # Track rewards
            reward_list.append(tot_reward)
            if (i + 1) % 100 == 0:
                ave_reward = np.mean(reward_list)
                ave_reward_list.append(ave_reward)
                reward_list = []
                print('Episode {} Average Reward: {}, epsilon:{}'.format(i + 1, ave_reward, epsilon))

        self.env.close()
        return ave_reward_list


def main():
    # Run Q-learning algorithm
    rewards = QLearner(0.2, 0.99).q_learning(0.02, 0, 5000)

    # Plot averaged rewards
    plt.plot(100 * (np.arange(len(rewards)) + 1), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.savefig('rewards.png')
    plt.close()


if __name__ == '__main__':
    main()

# sources : https://gist.github.com/gkhayes/3d154e0505e31d6367be22ed3da2e955  and  https://github.com/openai/gym/wiki/MountainCar-v0
