import math
import random
from collections import deque
from queue import Queue

import numpy as np
import tqdm

from script.util.Stacker import Stacker
from script.util.tensor_ops import *
from script.workbench.PERMemory import PERMemory
from script.workbench.gym_wrapper import gym_cartpole_v1_wrapper


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


class QNetwork:
    def __init__(self, input_size, output_size, lr, name='main', sess=None, dueling=True, PERMemory=True):
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.name = name
        self.sess = sess
        self.dueling = dueling
        self.PERMemory = PERMemory

        self.build_network()

    def build_network(self):
        with tf.variable_scope(self.name):
            self.x = placeholder(tf.float32, [-1, self.input_size], name='x')

            stack = Stacker(self.x)
            stack.linear(128)
            stack.relu()
            stack.linear(128)
            stack.relu()
            stack.linear(self.output_size)

            # proba
            if self.dueling:
                last = stack.last_layer
                self._value = linear(last, 1, name='value')
                self._advantage = linear(last, self.output_size, name='advantage')
                self._proba = self._value + self._advantage - tf.reduce_mean(self._advantage, reduction_indices=1,
                                                                             keep_dims=True)
            else:
                self._proba = stack.last_layer

            self._predict = tf.argmax(self._proba, axis=1)

            self.y = placeholder(tf.float32, [-1, self.output_size], 'Y')

            # loss
            if self.PERMemory:
                self.IS_weights = placeholder(tf.float32, [-1, ], name='IS_weights')
                loss = tf.reduce_mean((self._proba - self.y) * (self._proba - self.y), axis=1)
                self.loss = identity(self.IS_weights * loss, name='loss')

            else:
                self.loss = tf.squared_difference(self._proba, self.y, name='loss')

            self.loss_mean = tf.reduce_mean(self.loss, name='loss_mean')
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_mean)

    def predict(self, xs):
        return self.sess.run(self._predict, feed_dict={self.x: xs})

    def proba(self, xs):
        return self.sess.run(self._proba, feed_dict={self.x: xs})

    def train(self, xs, ys, IS_weights=None):
        # TODO
        if self.PERMemory:
            if IS_weights is None:
                raise TypeError('IS_weight is None')

            loss, _ = self.sess.run(
                [self.loss, self.train_op],
                feed_dict={
                    self.x: xs,
                    self.y: ys,
                    self.IS_weights: IS_weights,
                }
            )
        else:
            loss, _ = self.sess.run(
                [self.loss, self.train_op],
                feed_dict={
                    self.x: xs,
                    self.y: ys,
                }
            )
        return loss

    def loss(self, xs, ys, IS_weights=None, mean=True):
        # TODO
        if self.PERMemory:
            loss = self.sess.run(
                self.loss,
                feed_dict={
                    self.x: xs,
                    self.y: ys,
                    self.IS_weights: IS_weights,
                })
        else:
            loss = self.sess.run(self.loss, feed_dict={self.x: xs, self.y: ys})

        if mean:
            loss = np.mean(loss)

        return loss


class ReplayMemory:
    def __init__(self, max_size):
        self.max_size = max_size
        self._memory = deque()

    def __len__(self):
        return len(self._memory)

    def sample(self, n):
        return random.sample(self._memory, n)

    def store(self, item):
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
    def __init__(self, env, lr, discount_factor, state_space, action_space, sess=None, double=True, dueling=True,
                 replay_memory_size=100000, use_PERMemory=True, weight_copy_interval=10, train_interval=10,
                 batch_size=32):
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

        self.use_PERMemory = use_PERMemory
        self.replay_memory_size = replay_memory_size
        if self.use_PERMemory:
            self.__replay_memory = PERMemory(self.replay_memory_size)
        else:
            self.__replay_memory = ReplayMemory(self.replay_memory_size)

        self.double = double
        self.dueling = dueling
        self.weight_copy_interval = weight_copy_interval
        self.batch_size = batch_size
        self.train_interval = train_interval

        self.main_Qnet = QNetwork(state_space, action_space, self.lr, name='main', sess=self.sess, dueling=dueling)
        self.target_Qnet = QNetwork(state_space, action_space, self.lr, name='target', sess=self.sess, dueling=dueling)
        self.build_copy_weight()

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

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
            action = self.main_Qnet.predict(state)[0]

        return action

    def train_batch(self, batch, IS_weight=None):
        states, actions, next_states, rewards, dones = zip(*batch)
        states = np.concatenate(states)
        actions = np.array(actions)
        next_states = np.concatenate(next_states)
        rewards = np.array(rewards)
        dones = np.array(dones)

        # double
        if self.double:
            main_Q_predict = self.main_Qnet.predict(next_states)
            target_Q_proba = self.target_Qnet.proba(next_states)
            Q_next = target_Q_proba[np.arange(len(main_Q_predict)), main_Q_predict]
        else:
            Q_next = np.max(self.target_Qnet.proba(next_states), 1)

        X = states

        Q = rewards + self.discount_factor * Q_next * ~dones

        y = self.main_Qnet.proba(states)
        y[np.arange(len(X)), actions] = Q

        # Train our network using target and predicted Q values on each episode

        if self.use_PERMemory:
            loss = self.main_Qnet.train(X, y, IS_weight)
        else:
            loss = self.main_Qnet.train(X, y)

        return loss

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

                self.replay_memory.store((state, action, next_state, reward, done))

                reward_sum += 1

                state = next_state

                if done:
                    break

            # print(f'e: {self.e}')
            if episode % self.train_interval == 0 and len(self.replay_memory) >= self.batch_size:
                l_sum = 0.
                for i in range(50):
                    if self.use_PERMemory:
                        tree_idx, batch, IS_weights = self.replay_memory.sample(self.batch_size)
                        loss = self.train_batch(batch, IS_weights)
                        self.replay_memory.batch_update(tree_idx, loss)
                    else:
                        batch = self.replay_memory.sample(self.batch_size)
                        loss = self.train_batch(batch)
                    l_sum += np.mean(loss)
                tqdm.tqdm.write(f"loss: {l_sum / 50}")

            if episode % self.weight_copy_interval == 0:
                self.copy_main_to_target()

            reward_sums.append(reward_sum)
            ma_scalar.update(reward_sum)
            # bot_play_score = self.bot_play()
            # tqdm.tqdm.write(f"moving average reward sum : {ma_scalar.value()}, bot play score = {bot_play_score}")
            tqdm.tqdm.write(
                f"score:{reward_sum}, moving average score: {ma_scalar.value()}, max score:{max(reward_sums)}")

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
    env.env._max_episode_steps = 10000

    observation_space = 4
    action_space = 2

    lr = 0.01
    discount_factor = 0.99
    max_episodes = 3000
    # max_episodes = 30
    dueling = True
    double = True
    use_PERMemory = True
    weight_copy_interval = 10
    train_interval = 10
    batch_size = 32
    dqn = DQN(env, lr, discount_factor, observation_space, action_space, double=double, dueling=dueling,
              use_PERMemory=use_PERMemory, weight_copy_interval=weight_copy_interval, train_interval=train_interval,
              batch_size=batch_size)
    reward_sums = dqn.train(max_episodes)

    reward_sums = moving_average(reward_sums, 100)
    # from script.util.PlotTools import PlotTools
    # plot = PlotTools(save=False, show=True)
    # plot.plot(reward_sums)
    print(reward_sums)
    print("Score over time: " + str(sum(reward_sums) / max_episodes))
