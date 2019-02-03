from queue import Queue

import gym
import numpy as np
import tqdm


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


class Q_Learning:
    def __init__(self, env, lr, discount_factor, state_space, action_space, decay=1.):
        self.env = env
        self.lr = lr
        self.discount_factor = discount_factor
        self.state_space = state_space
        self.action_space = action_space
        self.table = np.zeros([state_space, action_space])
        self.decay = decay

    def update(self, state, action, next_state, reward):
        self.table[state, action] += self.lr * (
                reward + self.discount_factor * np.max(self.table[next_state, :]) - self.table[state, action])

    def chose(self, state, episode):
        decay_factor = (self.decay / (episode + 1))
        action = np.argmax(self.table[state, :] + np.random.randn(1, self.action_space) * decay_factor)

        return action

    def learn(self, n_episodes, max_move_count=99, n_windows=100):
        que = Queue(n_windows)
        windows_sum = 0

        reward_sums = []
        for episode in tqdm.trange(n_episodes):
            state = self.env.reset()

            reward_sum = 0
            for move_count in range(max_move_count):
                action = self.chose(state, episode)

                # env.step은 주어진 행동에 대한 다음 상태와 보상, 끝났는지 여부, 추가정보를 제공함
                next_state, reward, done, _ = self.env.step(action)

                self.update(state, action, next_state, reward)

                reward_sum += reward

                state = next_state

                if done:
                    # print(reward_sum)
                    break

            reward_sums.append(reward_sum)
            if que.full():
                v = que.get()
                windows_sum -= v
                que.put(reward_sum)
                windows_sum += reward_sum
            else:
                que.put(reward_sum)
                windows_sum += reward_sum
            windows_average = float(windows_sum) / n_windows
            tqdm.tqdm.write(f"moving average reward sum : {windows_average}")

        return reward_sums


def q_learning_frozenlake():
    is_slippery = True
    env = gym.make('FrozenLake-v0', )
    from gym.envs.registration import register
    register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': is_slippery},
        max_episode_steps=100,
        # reward_threshold=0.78,  # optimum = .8196
    )

    lr = .85
    discount_factor = .99
    num_episodes = 2000
    q_learning = Q_Learning(env, lr, discount_factor, env.observation_space.n, env.action_space.n)

    reward_sums = q_learning.learn(num_episodes)

    reward_sums = moving_average(reward_sums, 100)

    from script.util.PlotTools import PlotTools
    plot = PlotTools(save=False, show=True)
    plot.plot(reward_sums)
    # print(rList)
    print(q_learning.table)
    print("Score over time: " + str(sum(reward_sums) / num_episodes))


def q_learning_cartpole():
    env = gym.make('FrozenLake-v0', )
    # learn(env)
