from queue import Queue

import gym
import tensorflow as tf
import tqdm
import numpy as np


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


class env_wrapper:
    def __init__(self):
        pass

    @property
    def action_space(self):
        raise NotImplementedError

    @property
    def state_space(self):
        raise NotImplementedError

    @property
    def action_space_sample(self):
        raise NotImplementedError

    def reset(self):
        # return state
        raise NotImplementedError

    def step(self, action):
        # retun state, reward, done, info
        raise NotImplementedError


class gym_frozenlake_v0_wrapper(env_wrapper):
    def __init__(self, is_slippery=True):
        super().__init__()
        env = gym.make('FrozenLake-v0', )
        from gym.envs.registration import register
        register(
            id='FrozenLakeNotSlippery-v0',
            entry_point='gym.envs.toy_text:FrozenLakeEnv',
            kwargs={'map_name': '4x4', 'is_slippery': is_slippery},
            max_episode_steps=100,
            reward_threshold=0.78,  # optimum = .8196
        )
        self.__env = env

    @property
    def env(self):
        return self.__env

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def state_space(self):
        return self.env.observation_space

    def reset(self):
        observation = self.env.reset()

        observation = np.identity(16)[observation:observation + 1]

        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        observation = np.identity(16)[observation:observation + 1]

        return observation, reward, done, info


class gym_cartpole_v1_wrapper(env_wrapper):
    def __init__(self):
        super().__init__()
        self.__env = gym.make("CartPole-v1")

    @property
    def env(self):
        return self.__env

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def state_space(self):
        return self.env.observation_space

    @property
    def action_space_sample(self):
        return self.env.action_space.sample()

    def reset(self):
        state = self.env.reset()
        state = np.reshape(state, [1, len(state)])
        return state

    def step(self, action):
        s, r, d, info = self.env.step(action)

        s = np.reshape(s, [1, len(s)])
        return s, r, d, info


class QNetwork:
    def __init__(self, env, lr, discount_factor, state_space, action_space, sess=None):
        self.env = env
        self.lr = lr
        self.discount_factor = discount_factor
        self.state_space = state_space
        self.action_space = action_space

        self.sess = sess
        if self.sess is None:
            self.open()

        self._e = 1

        self.build_network()

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    @property
    def e(self):
        return self._e

    def open(self):
        self.sess = tf.Session()

    def close(self):
        if self.sess is not None:
            self.sess.close()
            self.sess = None

    def update(self, state, action, new_state, reward):
        Q_prime = self.sess.run(
            self.Qout,
            feed_dict={
                self.ph_state_space: new_state
            }
        )
        # 선택된 액션에 대한 타겟값과 maxq' 를 얻음
        maxQ1 = np.max(Q_prime)

        # target이 될 Q table을 업데이트 해줌
        targetQ = self.Q_S(state)
        targetQ[0, action] = reward + self.discount_factor * maxQ1

        # 예측된 Q 값과 target Q 값을 이용해 신경망을 학습함
        _, W1 = self.sess.run(
            [self.train_op, self.W],
            feed_dict={
                self.ph_state_space: state,
                self.nextQ: targetQ
            }
        )

    def decay_e(self, episode):
        self._e = 1. / ((episode / 50) + 10)

    def Q_S(self, state):
        return self.sess.run(
            self.Qout,
            feed_dict={
                self.ph_state_space: state
            }
        )

    def chose(self, state, env, random=True):
        a = self.sess.run(
            self.predict,
            feed_dict={
                self.ph_state_space: state
            }
        )

        if np.random.rand(1) < self._e and random:
            a[0] = env.action_space.sample()

        action = a[0]

        return action

    def build_network(self):
        self.ph_state_space = tf.placeholder(shape=[1, self.state_space], dtype=tf.float32)
        self.W = tf.Variable(tf.random_uniform([self.state_space, self.action_space], 0, 0.01))
        self.Qout = tf.matmul(self.ph_state_space, self.W)
        self.predict = tf.argmax(self.Qout, 1)

        self.nextQ = tf.placeholder(shape=[1, self.action_space], dtype=tf.float32)

        self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        self.train_op = trainer.minimize(self.loss)

    def learn(self, n_episodes, n_windows=100):
        que = Queue(n_windows)
        windows_sum = 0

        reward_sums = []
        for episode in tqdm.trange(n_episodes):
            state = self.env.reset()
            reward_sum = 0
            while True:
                action = self.chose(state, self.env)

                next_state, reward, done, _ = self.env.step(action)

                self.update(state, action, next_state, reward)

                reward_sum += reward

                state = next_state

                if done:
                    break

            self.decay_e(episode)

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


def qnetwork_frozenlake():
    env = gym_frozenlake_v0_wrapper()

    lr = 0.85
    discount_factor = 0.99
    max_episodes = 2000
    qnetwork = QNetwork(env, lr, discount_factor, 16, 4)
    reward_sums = qnetwork.learn(max_episodes)

    reward_sums = moving_average(reward_sums, 100)
    from script.util.PlotTools import PlotTools
    plot = PlotTools(save=False, show=True)
    plot.plot(reward_sums)
    # print(reward_sums)
    print("Score over time: " + str(sum(reward_sums) / max_episodes))


def qnetwork_cartpole():
    env = gym_cartpole_v1_wrapper()

    observation_space = 4
    action_space = 2

    lr = 0.85
    discount_factor = 0.99
    max_episodes = 2000
    qnetwork = QNetwork(env, lr, discount_factor, observation_space, action_space)
    reward_sums = qnetwork.learn(max_episodes)

    reward_sums = moving_average(reward_sums, 100)
    from script.util.PlotTools import PlotTools
    plot = PlotTools(save=False, show=True)
    plot.plot(reward_sums)
    print(reward_sums)
    print("Score over time: " + str(sum(reward_sums) / max_episodes))
