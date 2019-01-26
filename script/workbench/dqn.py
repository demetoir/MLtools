import math
from collections import deque
from queue import Queue
import random
import tqdm
import numpy as np

from script.util.Stacker import Stacker
from script.workbench.gym_wrapper import gym_frozenlake_v0_wrapper, gym_cartpole_v1_wrapper

from script.util.tensor_ops import *


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


class dqn_net:
    def __init__(self, input_size, output_size, lr, name='main', sess=None):
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.name = name
        self.sess = sess

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
            # stack.tanh()

            self._proba = stack.last_layer
            self._predict = tf.argmax(self._proba, axis=1)

            self.Y = placeholder(tf.float32, [-1, self.output_size], 'Y')

            self.loss = MSE_loss(self._proba, self.Y, 'loss')
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def predict(self, states):
        return self.sess.run(self._predict, feed_dict={self.x: states})

    def proba(self, states):
        return self.sess.run(self._proba, feed_dict={self.x: states})

    def update(self, xs, ys):
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.x: xs, self.Y: ys})
        return loss


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

        self.__replay_memory = deque()

        self.main_net = dqn_net(state_space, action_space, self.lr, name='main', sess=self.sess)
        self.target_net = dqn_net(state_space, action_space, self.lr, name='target', sess=self.sess)

        self.build_copy_weight()

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    @property
    def e(self):
        return self._e

    def decay_e(self, episode):
        t = episode
        val = 1.0 - math.log10((t + 1) * self.epsilon_decay)
        self._e = max(self.epsilon_min, min(float(self.e), val))

    def open(self):
        self.sess = tf.Session()

    def close(self):
        if self.sess is not None:
            self.sess.close()
            self.sess = None

    def update(self, batch):
        states, actions, next_states, rewards, dones = zip(*batch)
        states = np.concatenate(states)
        actions = np.array(actions)
        next_states = np.concatenate(next_states)
        rewards = np.array(rewards)
        dones = np.array(dones)
        # print(states.shape)
        # print(actions.shape)
        # print(next_states.shape)
        # print(rewards.shape)
        # print(dones.shape)

        X = states
        Q_next = np.max(self.target_net.proba(next_states), 1)

        Q = rewards + self.discount_factor * Q_next * ~dones

        y = self.main_net.proba(states)
        y[np.arange(len(X)), actions] = Q

        # Train our network using target and predicted Q values on each episode
        return self.main_net.update(X, y)

        # ys = []
        # xs = []
        # for state, action, new_state, reward, done in batch:
        #     Q = self.main_net.proba(state)
        #     Q_target = np.max(self.target_net.proba(new_state))
        #
        #     if done:
        #         Q[0, action] = reward
        #     else:
        #         Q[0, action] = reward + self.discount_factor * Q_target
        #
        #     xs += [state]
        #     ys += [Q]
        #
        # xs = np.concatenate(xs, axis=0)
        # ys = np.concatenate(ys, axis=0)
        #
        # return self.main_net.update(xs, ys)

    def chose(self, state, env, random=True):
        a = self.main_net.predict(state)
        # print(a)
        if np.random.rand(1) < self.e and random:
            a[0] = env.action_space.sample()

        action = a[0]

        return action

    def build_copy_weight(self):
        src_vars = collect_vars('main')
        dest_vars = collect_vars('target')

        ops = []
        for src, dest in zip(src_vars, dest_vars):
            ops += [dest.assign(src.value())]
        self.copy_ops = ops

    def copy_main_to_target(self):
        self.sess.run(self.copy_ops)

    @property
    def replay_memory(self):
        return self.__replay_memory

    def train(self, n_episodes, n_windows=10):
        que = Queue(n_windows)
        windows_sum = 0

        reward_sums = []
        for episode in tqdm.trange(n_episodes):
            self.decay_e(episode)

            state = self.env.reset()
            reward_sum = 0
            while True:
                action = self.chose(state, self.env)
                next_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = -200

                # print(f'action = {action}, r = {reward}')

                self.replay_memory.append((state, action, next_state, reward, done))
                if len(self.replay_memory) > 50000:
                    self.replay_memory.popleft()

                reward_sum += 1

                state = next_state

                if done:
                    break

            # print(f'e: {self.e}')
            if episode % 10 == 0:
                l_sum = 0.
                for i in range(50):
                    batch = random.sample(self.replay_memory, 10)

                    loss = self.update(batch)
                    l_sum += loss
                print(f"loss sum : {l_sum / 50}")

                self.copy_main_to_target()

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

    def bot_play(self):
        pass


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
    from script.util.PlotTools import PlotTools
    plot = PlotTools(save=False, show=True)
    plot.plot(reward_sums)
    print(reward_sums)
    print("Score over time: " + str(sum(reward_sums) / max_episodes))
