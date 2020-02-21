import gym
import numpy as np
import matplotlib.pyplot as plt
import math


class QLearner:
    def __init__(self, learning, discount):
        # Import and initialize Mountain Car Environment
        self.env = gym.make('MountainCar-v0')
        self.env.reset()
        # Initialize learning rate (alpha) and discount value (gamma)
        self.learning = learning
        self.discount = discount
        #######################################
        # Down-scaling feature space to discrete range, determine number of discrete states
        self.num_states = []
        x = self.env.observation_space.high - self.env.observation_space.low
        self.num_states.append(math.floor(x[0] * 20) + 1)
        self.num_states.append(math.floor(x[1] * 200) + 1)
        self.Q = np.random.uniform(low=-1, high=1, size=(self.num_states[0],
                                                         self.num_states[1], self.env.action_space.n))

    # Get discrete representation of state
    def discretize_state(self, state):
        # TODO: YOUR CODE
        x = state - self.env.observation_space.low
        disc_state = []
        disc_state.append(math.floor(x[0]*20))
        disc_state.append(math.floor(x[1] * 200))
        # print(disc_state)
        return disc_state

    # Determine next action using epsilon-greedy approach
    def get_action(self, state, epsilon):
        # TODO: YOUR CODE
        if (1-np.random.random()) < epsilon:
            action = np.random.randint(0, self.env.action_space.n)
        else:
            action = np.argmax(self.Q[state[0], state[1]])
        return action

    # Adjust Q value for current state
    def update_q(self, state, action, new_state, reward):
        # TODO: YOUR CODE
        # print(new_state[0],new_state[1])
        # print(self.Q[new_state[0], new_state[1]])
        delta = self.learning * (reward + (self.discount*np.max(self.Q[new_state[0], new_state[1]]))-self.Q[state[0], state[1], action])
        return delta
    # Q-Learning function
    def q_learning(self, epsilon, min_eps, episodes):
        # Initialize variables to track rewards
        reward_list = []
        ave_reward_list = []

        # Run Q-learning algorithm
        for i in range(episodes):
            # Initialize parameters
            done = False
            tot_reward, reward = 0, 0
            state = self.env.reset()
            disc_state = self.discretize_state(state)
            # print(disc_state)

            while not done:
                # Render environment for last twenty episodes
                if i >= (episodes - 20):
                    self.env.render()

                # Determine next action, state, and reward
                # TODO: YOUR CODE
                A = self.get_action(disc_state, epsilon)

                new_state, reward, done, info = self.env.step(A)

                disc_new_state = self.discretize_state(new_state)
                # print(disc_new_state)
                # Update Q
                if done and new_state[0] >= 0.5:
                    self.Q[disc_state[0], disc_state[1], A] = reward
                else:
                    self.Q[disc_state[0], disc_state[1], A] += self.update_q(disc_state, A, disc_new_state, reward)
                # Update variables
                # TODO: YOUR CODE
                tot_reward += reward
                disc_state = disc_new_state
            # Decay epsilon
            # TODO: YOUR CODE
            if epsilon > min_eps:
                epsilon -= (epsilon - min_eps) / episodes
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
