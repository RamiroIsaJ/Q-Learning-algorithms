import numpy as np
import tensorflow as tf
import pandas as pd
from inter_maze import Maze


np.random.seed(1)
tf.random.set_seed(1)
tf.compat.v1.reset_default_graph()


class QNet:
    def __init__(self, n_states, n_actions, learning_rate=0.01, e_greedy=1.0, reward_decay=0.8, memory_size=2000,
                 batch_size=100,
                 sess=None):
        self.lr = learning_rate
        self.gamma = reward_decay
        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon = e_greedy
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory_counter = 0
        self.memory = np.zeros((self.memory_size, self.n_states*2+2))
        self.s = tf.placeholder(tf.float32, [None, self.n_states])
        self.pred_Q = tf.placeholder(tf.float32, [None, self.n_actions])
        self.eval_Q = tf.placeholder(tf.float32, [None, self.n_actions])
        self.q_table = pd.DataFrame(columns=list(range(1)), dtype=np.float64)
        self.build_net()

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

    def build_layers(self):
        n_l1 = 40
        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1', [self.n_states, n_l1], initializer=tf.random_normal_initializer(0., 0.3))
            b1 = tf.get_variable('b1', [1, n_l1], initializer=tf.constant_initializer(0.1))
            l1 = tf.nn.relu(tf.matmul(self.s, w1)+b1)
        with tf.variable_scope('l2'):
            w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=tf.random_normal_initializer(0., 0.3))
            b2 = tf.get_variable('b2', [1, self.n_actions], initializer=tf.constant_initializer(0.1))
            out = tf.nn.relu(tf.matmul(l1, w2)+b2)
        return out

    def build_net(self):
        with tf.variable_scope('eval'):
            self.eval_Q = self.build_layers()
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.pred_Q, self.eval_Q))
            # self.loss = tf.reduce_sum(tf.square(self.pred_Q - self.eval_Q))
        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            # self.train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.loss)

    def choose_action(self, observation):
        self.check_state(observation)
        observation = self.q_table.loc[observation, 0]
        self.action_value = self.sess.run(self.eval_Q, feed_dict={self.s: observation})

        if np.random.uniform() < self.epsilon:
            action = np.asarray([np.random.randint(0, self.n_actions)])
        else:
            action = self.sess.run(tf.argmax(self.action_value, 1))

        return action

    def check_state(self, state):
        if state not in self.q_table.index:
            p1 = self.q_table.shape[0]
            self.q_table = self.q_table.append(pd.Series([np.identity(16)[p1:p1+1]],
                                                         index=self.q_table.columns, name=state))

    def learn(self, ep, s, a, r, s_):
        self.check_state(s_)
        ob = self.q_table.loc[s, 0]
        target_q = self.action_value.copy()

        ob_ = self.q_table.loc[s_, 0]
        q_next = self.sess.run([self.eval_Q], feed_dict={self.s: ob_})
        max_q = np.max(q_next)
        print(q_next)

        if s_ != 'terminal':
            target_q[0, a[0]] = r + self.gamma * max_q
        else:
            target_q[0, a[0]] = r

        self.sess.run([self.train_op], feed_dict={self.s: ob, self.pred_Q: target_q})

        # update epsilon value according to the Deep Mind Google
        if self.epsilon > 0.1:
            self.epsilon -= (1/ep)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn_memory(self, ep, s, a, r, s_):
        self.check_state(s_)
        ob = self.q_table.loc[s, 0]
        ob_ = self.q_table.loc[s_, 0]
        self.store_transition(ob[0], a, r, ob_[0])

        if self.memory_counter % self.memory_size == 0:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]

            q_next = self.sess.run(self.eval_Q, {self.s: batch_memory[:, -self.n_states:]})
            q_eval = self.sess.run(self.eval_Q, {self.s: batch_memory[:, :self.n_states]})

            target_q = q_eval.copy()
            max_q = np.max(q_next, axis=1)

            # actions vector
            eval_act_index = batch_memory[:, self.n_states].astype(int)
            # reward vector
            reward = batch_memory[:, self.n_states + 1]
            # index where reward is negative
            index_neg = np.argwhere(reward < 0)
            index_pos = np.argwhere(reward > 0)

            # update target values
            target_q[index_pos, eval_act_index[index_pos]] = reward[index_pos] + self.gamma * max_q[index_pos]
            target_q[index_neg, eval_act_index[index_neg]] = reward[index_neg]

            self.sess.run([self.train_op], feed_dict={self.s: batch_memory[:, :self.n_states], self.pred_Q: target_q})

        else:
            target_q = self.action_value.copy()
            q_next = self.sess.run([self.eval_Q], feed_dict={self.s: ob_})
            max_q = np.max(q_next)

            if s_ != 'terminal':
                target_q[0, a[0]] = r + self.gamma * max_q
            else:
                target_q[0, a[0]] = r

            self.sess.run([self.train_op], feed_dict={self.s: ob, self.pred_Q: target_q})
        '''
        # update epsilon value according to the Deep Mind Google
        if self.epsilon > 0.1:
            self.epsilon -= (1/ep)
        '''


class DQNet:
    def __init__(self, n_states, n_actions, learning_rate=0.01, e_greedy=1.0, reward_decay=0.9, memory_size=300,
                 batch_size=100, replace_tar_iter=100, double_q=True, sess=None, output_graph=True):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.epsilon = e_greedy
        self.gamma = reward_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.replace_tar_iter = replace_tar_iter
        self.double_q = double_q
        self.memory_counter, self.learn_counter = 0, 0
        self.memory = np.zeros((self.memory_size, self.n_states * 2 + 2))
        self.s = tf.placeholder(tf.float32, [None, self.n_states], name='s')
        self.s_ = tf.placeholder(tf.float32, [None, self.n_states], name='s_')
        self.pred_Q = tf.placeholder(tf.float32, [None, self.n_actions])
        self.eval_Q = tf.placeholder(tf.float32, [None, self.n_actions])
        self.tar_Q = tf.placeholder(tf.float32, [None, self.n_actions])
        self.loss = tf.placeholder(tf.float32, [None, None])
        self.train_net = tf.placeholder(tf.float32, [None, None])
        self.q_table = pd.DataFrame(columns=list(range(1)), dtype=np.float64)
        self.build_nets()

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        e_params = tf.get_collection('tar_net_params')
        t_params = tf.get_collection('eval_net_params')
        self.replace_tar = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

    def build_layers(self, s, nl, w_initial, b_initial, c_name):
        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1', [self.n_states, nl], initializer=w_initial, collections=c_name)
            b1 = tf.get_variable('b1', [1, nl], initializer=b_initial, collections=c_name)
            l1 = tf.nn.relu(tf.matmul(s, w1)+b1)

        with tf.variable_scope('l2'):
            w2 = tf.get_variable('w2', [nl, self.n_actions], initializer=w_initial, collections=c_name)
            b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initial, collections=c_name)
            out_net = tf.matmul(l1, w2)+b2
        return out_net

    def build_nets(self):
        with tf.variable_scope('eval_net'):
            c_names, nl = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 30
            w_initial, b_initial = tf.random_normal_initializer(0.0, 0.3), tf.constant_initializer(0.1)
            self.eval_Q = self.build_layers(self.s, nl, w_initial, b_initial, c_names)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.eval_Q, self.pred_Q))

        with tf.variable_scope('train'):
            self.train_net = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('target_net'):
            c_names = ['tar_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            w_initial = tf.random_normal_initializer(0.0, 0.3)
            self.tar_Q = self.build_layers(self.s_, nl, w_initial, b_initial, c_names)

    def choose_action(self, observation):
        self.check_state(observation)
        observation = self.q_table.loc[observation, 0]
        action_value = self.sess.run(self.eval_Q, feed_dict={self.s: observation})

        if np.random.uniform() < self.epsilon:
            action = np.asarray([np.random.randint(0, self.n_actions)])
        else:
            action = self.sess.run(tf.argmax(action_value, 1))

        return action

    def check_state(self, state):
        if state not in self.q_table.index:
            p1 = self.q_table.shape[0]
            self.q_table = self.q_table.append(pd.Series([np.identity(16)[p1:p1 + 1]],
                                                         index=self.q_table.columns, name=state))

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self, ep, s, a, r, s_):
        self.check_state(s_)
        ob = self.q_table.loc[s, 0]
        ob_ = self.q_table.loc[s_, 0]
        self.store_transition(ob[0], a, r, ob_[0])

        if self.learn_counter % self.replace_tar_iter == 0:
            self.sess.run(self.replace_tar)
            print('\n Replace target parameters \n')

        if self.memory_counter > self.memory_size:
            idx_sample = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            idx_sample = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[idx_sample, :]

        next_q = self.sess.run(self.tar_Q, feed_dict={self.s_: batch_memory[:, -self.n_states:]})
        eval_next_q = self.sess.run(self.eval_Q, feed_dict={self.s: batch_memory[:, -self.n_states:]})

        eval_q = self.sess.run(self.eval_Q, feed_dict={self.s: batch_memory[:, :self.n_states]})

        target_q = eval_q.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_states].astype(int)
        reward = batch_memory[:, self.n_states + 1]
        # index where reward is negative
        index_neg = np.argwhere(reward < 0)
        index_pos = np.argwhere(reward > 0)

        if self.double_q:
            max_act4next = np.argmax(eval_next_q, axis=1)
            selected_q_next = next_q[batch_index, max_act4next]
        else:
            selected_q_next = np.max(next_q, axis=1)

        target_q[index_pos, eval_act_index[index_pos]] = reward[index_pos] + self.gamma * selected_q_next[index_pos]
        target_q[index_neg, eval_act_index[index_neg]] = reward[index_neg]

        self.sess.run([self.train_net], feed_dict={self.s: batch_memory[:, :self.n_states], self.pred_Q: target_q})

        # update epsilon value according to the Deep Mind Google
        if self.epsilon > 0.1:
            self.epsilon -= (1/ep)

        self.learn_counter += 1


def updateQN():
    epochs = 500
    for episode in range(epochs):
        observation = env.reset()

        while True:
            env.render()

            action = RL1.choose_action(str(observation))

            observation_, reward, done = env.step(action)

            RL1.learn_memory(epochs, str(observation), action, reward, str(observation_))

            observation = observation_

            if done:
                break
        print('Finish episode')

    print('Game over')
    env.destroy()


def updateDQN():
    epochs = 500
    for episode in range(epochs):
        observation = env.reset()

        while True:
            env.render()

            action = RL2.choose_action(str(observation))

            observation_, reward, done = env.step(action)

            RL2.learn(epochs, str(observation), action, reward, str(observation_))

            observation = observation_

            if done:
                break
        print('Finish episode')

    print('Game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL1 = QNet(n_states=16, n_actions=4)
    RL2 = DQNet(n_actions=4, n_states=16)
    env.after(100, updateDQN())
    env.mainloop()
