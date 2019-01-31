import math
import random
from collections import deque
from pprint import pprint
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
            stack.linear(64)
            stack.relu()
            stack.linear(64)
            stack.relu()
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

            if self.PERMemory:
                self.IS_weights = placeholder(tf.float32, [-1, ], name='IS_weights')
                loss = tf.reduce_mean((self._proba - self.y) * (self._proba - self.y), axis=1)

                self.loss = identity(self.IS_weights * loss, name='loss')
                self.loss_mean = tf.reduce_mean(self.loss, name='loss_mean')

                self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_mean)
            else:
                self.loss = MSE_loss(self._proba, self.y, 'loss')

                self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def predict(self, xs):
        return self.sess.run(self._predict, feed_dict={self.x: xs})

    def proba(self, xs):
        return self.sess.run(self._proba, feed_dict={self.x: xs})

    def train(self, xs, ys, IS_weights=None):
        if self.PERMemory:
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

    def loss(self, xs, ys):
        return self.sess.run(self.loss, feed_dict={self.x: xs, self.y: ys})


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


class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # Number of leaf nodes (final nodes) that contains experiences

        self.tree = np.zeros(2 * capacity - 1)

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        # self.data_pointer = (self.data_pointer + 1) % self.leaf_size

    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        diff = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the diff through tree
        while tree_index != 0:  # this method is faster than the recursive loop in the reference code
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += diff

    def get_leaf(self, v):
        parent_index = 0
        while True:  # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            if v <= self.tree[left_child_index]:
                parent_index = left_child_index
            else:
                v -= self.tree[left_child_index]
                parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]


class PERMemory(object):  # stored as ( s, a, r, s_ ) in SumTree
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity, e=0.01, a=0.6, b=0.4):
        self.e = e
        self.a = a
        self.b = b
        self.tree = SumTree(capacity)
        self.cnt = 0

    @property
    def capacity(self):
        return self.tree.capacity

    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)  # set the max p for new p

        self.cnt = min(self.cnt + 1, self.capacity)

    def __len__(self):
        return self.cnt

    def sample(self, n):
        # Create a sample array that will contains the minibatch
        memory_b = []

        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n  # priority segment

        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1

        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        # p_min = np.max(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        # print(self.tree.total_priority)
        # max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)

            # P(j)
            sampling_probabilities = priority / self.tree.total_priority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            # b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b) / max_weight
            b_ISWeights[i, 0] = np.power((sampling_probabilities * self.capacity), -self.PER_b)


            b_idx[i] = index

            experience = [data]

            memory_b.append(experience)

        memory_b = np.array(memory_b).reshape([32, 5])
        b_ISWeights = np.array(b_ISWeights).reshape([-1])
        b_ISWeights /= np.max(b_ISWeights)
        # print(b_ISWeights)
        return b_idx, memory_b, b_ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


# def full_with_memory():
#     # Instantiate memory
#     memory = PERMemory(memory_size)
#
#     # Render the environment
#     game.new_episode()
#
#     for i in range(pretrain_length):
#         # If it's the first step
#         if i == 0:
#             # First we need a state
#             state = game.get_state().screen_buffer
#             state, stacked_frames = stack_frames(stacked_frames, state, True)
#
#         # Random action
#         action = random.choice(possible_actions)
#
#         # Get the rewards
#         reward = game.make_action(action)
#
#         # Look if the episode is finished
#         done = game.is_episode_finished()
#
#         # If we're dead
#         if done:
#             # We finished the episode
#             next_state = np.zeros(state.shape)
#
#             # Add experience to memory
#             # experience = np.hstack((state, [action, reward], next_state, done))
#
#             experience = state, action, reward, next_state, done
#             memory.store(experience)
#
#             # Start a new episode
#             game.new_episode()
#
#             # First we need a state
#             state = game.get_state().screen_buffer
#
#             # Stack the frames
#             state, stacked_frames = stack_frames(stacked_frames, state, True)
#
#         else:
#             # Get the next state
#             next_state = game.get_state().screen_buffer
#             next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
#
#             # Add experience to memory
#             experience = state, action, reward, next_state, done
#             memory.store(experience)
#
#             # Our state is now the next_state
#             state = next_state
#     pass


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
                 replay_memory_size=100000, PER=True):
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

        self.PER = PER
        self.replay_memory_size = replay_memory_size
        if self.PER:
            self.__replay_memory = PERMemory(self.replay_memory_size)
        else:
            self.__replay_memory = ReplayMemory(self.replay_memory_size)

        self.double = double
        self.dueling = dueling
        self.weight_copy_interval = 10
        self.batch_size = 32
        self.train_interval = 10

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

        if self.PER:
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
                    if self.PER:
                        tree_idx, batch, IS_weights = self.replay_memory.sample(self.batch_size)

                        loss = self.train_batch(batch, IS_weights)
                        self.replay_memory.batch_update(tree_idx, loss)
                    else:
                        batch = self.replay_memory.sample(self.batch_size)
                        loss = self.train_batch(batch)
                    l_sum += loss
                tqdm.tqdm.write(f"loss: {l_sum / 50}")

            if episode % self.weight_copy_interval == 0:
                self.copy_main_to_target()

            reward_sums.append(reward_sum)
            ma_scalar.update(reward_sum)
            # bot_play_score = self.bot_play()
            # tqdm.tqdm.write(f"moving average reward sum : {ma_scalar.value()}, bot play score = {bot_play_score}")
            tqdm.tqdm.write(f"moving average reward sum : {ma_scalar.value()}, max score = {max(reward_sums)}")

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
    dueling = True
    double = True
    PER = True
    dqn = DQN(env, lr, discount_factor, observation_space, action_space, double=double, dueling=dueling, PER=PER)
    reward_sums = dqn.train(max_episodes)

    reward_sums = moving_average(reward_sums, 100)
    # from script.util.PlotTools import PlotTools
    # plot = PlotTools(save=False, show=True)
    # plot.plot(reward_sums)
    print(reward_sums)
    print("Score over time: " + str(sum(reward_sums) / max_episodes))
