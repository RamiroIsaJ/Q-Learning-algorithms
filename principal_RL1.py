from inter_maze import MazeRL as maze
from RL_Q_SARSA import QLearningTable
from RL_Q_SARSA import SarsaTable
from RL_Q_SARSA import SarsaLambdaTable


def updateQ():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()
            # RL choose action based in observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from transition
            RL.learn(str(observation), action, reward, str(observation_))

            print('Number of Episode:  ' + str(episode) + ' ===>  Reward:  ' + str(reward))

            # swap observation
            observation = observation_

            if done:
                break
    print('Learning finish')
    env.destroy()


def updateS():
    for episode in range(100):
        observation = env.reset()

        action = RL1.choose_action(str(observation))

        while True:
            env.render()

            observation_, reward, done = env.step(action)

            action_ = RL1.choose_action(str(observation_))

            RL1.learn(str(observation), action, reward, str(observation_), action_)

            print('Number of Episode:  ' + str(episode) + ' ===>  Reward:  ' + str(reward))

            observation = observation_
            action = action_

            if done:
                break
    print('Learning finish')
    env.destroy()


def updateSL():
    for episode in range(100):
        observation = env.reset()

        action = RL1.choose_action(str(observation))
        RL2.eligibility_trace *= 0

        while True:
            env.render()

            observation_, reward, done = env.step(action)

            action_ = RL2.choose_action(str(observation_))

            RL2.learn(str(observation), action, reward, str(observation_), action_)

            print('Number of Episode:  ' + str(episode) + ' ===>  Reward:  ' + str(reward))

            observation = observation_
            action = action_

            if done:
                break

    print('Learning finish')
    env.destroy()


if __name__ == "__main__":
    env = maze()

    RL = QLearningTable(actions=list(range(env.n_actions)))
    RL1 = SarsaTable(actions=list(range(env.n_actions)))
    RL2 = SarsaLambdaTable(actions=list(range(env.n_actions)))

    env.after(100, updateQ())
    # env.after(100, updateS())
    # env.after(100, updateSL())
    env.mainloop()




