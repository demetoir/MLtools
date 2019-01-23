from queue import Queue

import gym
import numpy as np
import tensorflow as tf
import tqdm


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

        Q_prime = self.sess.run(self.Qout, feed_dict={self.inputs1: np.identity(16)[new_state:new_state + 1]})
        # 선택된 액션에 대한 타겟값과 maxq' 를 얻음
        maxQ1 = np.max(Q_prime)

        # target이 될 Q table을 업데이트 해줌
        targetQ = self.Q_S(state)
        targetQ[0, action] = reward + self.discount_factor * maxQ1

        # 예측된 Q 값과 target Q 값을 이용해 신경망을 학습함
        _, W1 = self.sess.run(
            [self.train_op, self.W],
            feed_dict={
                self.inputs1: np.identity(16)[state:state + 1],
                self.nextQ: targetQ
            }
        )

    def decay_e(self, episode):
        self._e = 1. / ((episode / 50) + 10)

    def Q_S(self, state):
        allQ = self.sess.run(self.Qout, feed_dict={self.inputs1: np.identity(16)[state:state + 1]})

        return allQ

    def chose(self, state, env, random=True):
        a = self.sess.run(self.predict, feed_dict={self.inputs1: np.identity(16)[state:state + 1]})

        if np.random.rand(1) < self._e and random:
            a[0] = env.action_space.sample()

        action = a[0]

        return action

    def build_network(self):
        self.inputs1 = tf.placeholder(shape=[1, 16], dtype=tf.float32)
        self.W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))

        self.Qout = tf.matmul(self.inputs1, self.W)
        self.predict = tf.argmax(self.Qout, 1)

        self.nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32)

        self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        self.train_op = trainer.minimize(self.loss)

    def learn(self, n_episodes, max_move_count=99, n_windows=100):
        que = Queue(n_windows)
        windows_sum = 0

        reward_sums = []
        for episode in tqdm.trange(n_episodes):
            state = self.env.reset()

            reward_sum = 0
            for move_count in range(max_move_count):
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


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def qnetwork_frozenlake():
    is_slippery = True
    env = gym.make('FrozenLake-v0', )
    from gym.envs.registration import register
    register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': is_slippery},
        max_episode_steps=100,
        reward_threshold=0.78,  # optimum = .8196
    )

    lr = 0.85
    discount_factor = 0.99
    max_episodes = 2000
    qnetwork = QNetwork(env, lr, discount_factor, env.observation_space.n, env.action_space.n)
    reward_sums = qnetwork.learn(max_episodes)

    reward_sums = moving_average(reward_sums, 100)
    from script.util.PlotTools import PlotTools
    plot = PlotTools(save=False, show=True)
    plot.plot(reward_sums)
    # print(reward_sums)
    print("Score over time: " + str(sum(reward_sums) / max_episodes))


def qnetwork_cartpole():
    env = gym.make('CartPole-v0')

    lr = 0.85
    discount_factor = 0.99
    max_episodes = 2000
    qnetwork = QNetwork(env, lr, discount_factor, env.observation_space.n, env.action_space.n)
    reward_sums = qnetwork.learn(max_episodes)

    reward_sums = moving_average(reward_sums, 100)
    from script.util.PlotTools import PlotTools
    plot = PlotTools(save=False, show=True)
    plot.plot(reward_sums)
    # print(reward_sums)
    print("Score over time: " + str(sum(reward_sums) / max_episodes))
