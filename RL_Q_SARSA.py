import numpy as np
import pandas as pd


class RL(object):

    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space
        self.alpha = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
             pd.Series([0]*len(self.actions), index=self.q_table.columns, name=state)
             )

    def choose_action(self, observation):
        self.check_state_exist(observation)
        print('--------------- Q-TABLE ------------------')
        print(self.q_table)
        print('------------------------------------------')
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            action = np.random.choice(self.actions)

        return action


# off policy
class QLearningTable(RL):
    def __init__(self, actions,  learning_rate=0.01, reward_decay=0.9, e_greedy=0.8):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            # next state is terminal
            q_target = r

        print('Value Function:     ' + str(q_target))

        # update table
        self.q_table.loc[s, a] += self.alpha * (q_target - q_predict)


# on policy
class SarsaTable(RL):
    def __init__(self, actions,  learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma*self.q_table.loc[s_, a_]
        else:
            q_target = r

        print('Value Function:     ' + str(q_target))

        self.q_table.loc[s, a] += self.alpha * (q_target - q_predict)


# backward eligibility traces
class SarsaLambdaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            n_state = pd.Series([0]*len(self.actions), index=self.q_table.columns, name=state)

            self.q_table = self.q_table.append(n_state)

            # update eligibility trace
            self.eligibility_trace = self.eligibility_trace.append(n_state)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r

        # increase trace amount for visit state-action pair
        # Method 1
        # self.eligibility_trace.loc[s, a] += 1

        # Method 2
        self.eligibility_trace.loc[s, :] *= 0
        self.eligibility_trace.loc[s, a] = 1

        print('Value Function:     ' + str(q_target))

        # Q update
        self.q_table += self.alpha * (q_target - q_predict) * self.eligibility_trace

        # decay eligibility trace after update
        self.eligibility_trace *= self.gamma * self.lambda_



