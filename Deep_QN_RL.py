from inter_maze import MazeRL as maze
from Deep_Q_network_2 import DoubleDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


env = maze()
MEMORY_SIZE = 300
ACTION_SPACE = 4

sess = tf.compat.v1.Session()
with tf.compat.v1.variable_scope('Natural_DQN'):
    natural_DQN = DoubleDQN(n_actions=ACTION_SPACE, n_features=4, memory_size=MEMORY_SIZE, e_greedy_increment=0.001,
                            double_q=False, sess=sess)

with tf.compat.v1.variable_scope('Double_DQN'):
    double_DQN = DoubleDQN(n_actions=ACTION_SPACE, n_features=4, memory_size=MEMORY_SIZE, e_greedy_increment=0.001,
                           double_q=True, sess=sess)

sess.run(tf.compat.v1.global_variables_initializer())


def train(RLe):
    total_steps = 0
    for epochs in range(20000):

        observation = env.reset()

        while True:
            env.render()
            observation = np.asarray(observation)

            action = RLe.choose_action(observation)

            observation_, reward, done = env.step(action)
            observation_ = np.asarray(observation_)

            RLe.store_transition(observation, action, reward, observation_)

            print('Number of Epoch:  ' + str(epochs) + ' ===>  Reward:  ' + str(reward))

            if total_steps > MEMORY_SIZE:
                print('Training Neural Network ...')
                RLe.learn()
                total_steps = 0

            if done:
                break

            observation = observation_
            total_steps += 1

    return RLe.q


# q_natural = train(natural_DQN)
q_double = train(double_DQN)

# plt.plot(np.array(q_natural), c='r', label='natural')
plt.plot(np.array(q_double), c='b', label='double')
plt.legend(loc='best')
plt.ylabel('Q eval')
plt.xlabel('training steps')
plt.grid()
plt.show()


