import math
import random
from collections import deque
from queue import Queue

import numpy as np
import tqdm

from script.util.Stacker import Stacker
from script.util.tensor_ops import *
from script.workbench.gym_wrapper import gym_cartpole_v1_wrapper


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


class QNetwork:
    def __init__(self, input_size, output_size, lr, name='main', sess=None):
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.name = name
        self.sess = sess
        self.dueling = True

        self.build_network()

    def build_network(self):
        with tf.variable_scope(self.name):
            self.x = placeholder(tf.float32, [-1, self.input_size], name='x')
            stack = Stacker(self.x)
            stack.linear(24)
            stack.tanh()
            stack.linear(48)
            stack.tanh()
            stack.linear(self.output_size)

            if self.dueling:
                last = stack.last_layer
                self._value = linear(last, 1, name='value')
                self._advantage = linear(last, self.output_size, name='advantage')
                self._proba = self._value + self._advantage - tf.reduce_mean(self._advantage, reduction_indices=1,
                                                                             keep_dims=True)
                self._predict = tf.argmax(self._proba, axis=1)
            else:
                self._proba = stack.last_layer
                self._predict = tf.argmax(self._proba, axis=1)

            self.y = placeholder(tf.float32, [-1, self.output_size], 'Y')

            self.loss = MSE_loss(self._proba, self.y, 'loss')
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def predict(self, xs):
        return self.sess.run(self._predict, feed_dict={self.x: xs})

    def proba(self, xs):
        return self.sess.run(self._proba, feed_dict={self.x: xs})

    def train(self, xs, ys):
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.x: xs, self.y: ys})
        return loss

    def loss(self, xs, ys):
        return self.sess.run(self.loss, feed_dict={self.x: xs, self.y: ys})


class replay_memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self._memory = deque()

    def __len__(self):
        return len(self._memory)

    def sample(self, n):
        return random.sample(self._memory, n)

    def append(self, item):
        self._memory.append(item)
        if len(self._memory) > self.max_size:
            self._memory.popleft()


class moving_average_scalar:
    def __init__(self, window_size, name='moving average'):
        self.window_size = window_size
        self.que = Queue(self.window_size)
        self.name = name
        self.windows_sum = 0.

    def init_sum(self):
        self.windows_sum = 0.

    def update(self, item):
        if self.que.full():
            v = self.que.get()
            self.windows_sum -= v
            self.que.put(item)
            self.windows_sum += item
        else:
            self.que.put(item)
            self.windows_sum += item

    def value(self):
        return self.windows_sum / self.window_size


class DQN:
    def __init__(self, env, lr, discount_factor, state_space, action_space, sess=None):
        self.env = env
        self.lr = lr
        self.discount_factor = discount_factor
        self.state_space = state_space
        self.action_space = action_space

        self.sess = sess
        if self.sess is None:
            self.open()

        self._e = 1.
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.__replay_memory = replay_memory(100000)

        self.main_qnet = QNetwork(state_space, action_space, self.lr, name='main', sess=self.sess)
        self.target_qnet = QNetwork(state_space, action_space, self.lr, name='target', sess=self.sess)

        self.build_copy_weight()

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.double = True

    @property
    def e(self):
        return self._e

    def decay_e(self, episode):
        t = episode
        val = 1.0 - math.log10((t + 1) * self.epsilon_decay)
        self._e = max(self.epsilon_min, min(float(self.e), val))

    @property
    def replay_memory(self):
        return self.__replay_memory

    def open(self):
        self.sess = tf.Session()

    def close(self):
        if self.sess is not None:
            self.sess.close()
            self.sess = None

    def build_copy_weight(self):
        src_vars = collect_vars('main')
        dest_vars = collect_vars('target')

        ops = []
        for src, dest in zip(src_vars, dest_vars):
            ops += [dest.assign(src.value())]
        self.copy_ops = tf.group(ops, name='copy_network_op')

    def copy_main_to_target(self):
        self.sess.run(self.copy_ops)

    def chose(self, state, env, random=True):
        if np.random.rand(1) < self.e and random:
            action = env.action_space.sample()
        else:
            action = self.main_qnet.predict(state)[0]

        return action

    def train_batch(self, batch):
        states, actions, next_states, rewards, dones = zip(*batch)
        states = np.concatenate(states)
        actions = np.array(actions)
        next_states = np.concatenate(next_states)
        rewards = np.array(rewards)
        dones = np.array(dones)

        # double
        if self.double:
            main_Q_predict = self.main_qnet.predict(next_states)
            # print(main_Q_predict.shape)
            # print(main_Q_predict)

            target_Q_proba = self.target_qnet.proba(next_states)
            # print(target_Q_proba.shape)
            # print(target_Q_proba)
            Q_next = target_Q_proba[np.arange(len(main_Q_predict)), main_Q_predict]
            # print(Q_next)

        else:
            Q_next = np.max(self.target_qnet.proba(next_states), 1)

        X = states

        # Q_next = np.max(self.target_net.proba(next_states), 1)

        Q = rewards + self.discount_factor * Q_next * ~dones

        y = self.main_qnet.proba(states)
        y[np.arange(len(X)), actions] = Q

        # Train our network using target and predicted Q values on each episode
        return self.main_qnet.train(X, y)

    def train(self, n_episodes, n_windows=10):
        ma_scalar = moving_average_scalar(n_windows)

        reward_sums = []
        for episode in tqdm.trange(n_episodes):
            self.decay_e(episode)

            state = self.env.reset()
            reward_sum = 0
            while True:
                action = self.chose(state, self.env)
                next_state, reward, done, _ = self.env.step(action)

                # if done:
                #     reward = -200

                self.replay_memory.append((state, action, next_state, reward, done))

                reward_sum += 1

                state = next_state

                if done:
                    break

            # print(f'e: {self.e}')
            if episode % 10 == 0:
                l_sum = 0.
                for i in range(50):
                    batch = self.replay_memory.sample(10)

                    loss = self.train_batch(batch)
                    l_sum += loss
                print(f"loss sum : {l_sum / 50}")

                self.copy_main_to_target()

            reward_sums.append(reward_sum)
            ma_scalar.update(reward_sum)
            # bot_play_score = self.bot_play()
            # tqdm.tqdm.write(f"moving average reward sum : {ma_scalar.value()}, bot play score = {bot_play_score}")
            tqdm.tqdm.write(f"moving average reward sum : {ma_scalar.value()}")

            self.bot_play()

        return reward_sums

    def bot_play(self):
        state = self.env.reset()
        reward_sum = 0
        while True:
            action = self.chose(state, self.env)

            next_state, reward, done, _ = self.env.step(action)
            reward_sum += reward
            state = next_state

            if done:
                break

        return reward_sum


def dqn_cartpole():
    env = gym_cartpole_v1_wrapper()

    observation_space = 4
    action_space = 2

    lr = 0.01
    discount_factor = 0.99
    max_episodes = 3000
    # max_episodes = 30
    dqn = DQN(env, lr, discount_factor, observation_space, action_space)
    reward_sums = dqn.train(max_episodes)

    reward_sums = moving_average(reward_sums, 100)
    # from script.util.PlotTools import PlotTools
    # plot = PlotTools(save=False, show=True)
    # plot.plot(reward_sums)
    print(reward_sums)
    print("Score over time: " + str(sum(reward_sums) / max_episodes))
