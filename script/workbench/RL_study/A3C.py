import threading
import time

import gym
import numpy as np

from script.util.Stacker import Stacker
from script.util.tensor_ops import *

# global variables for threading
episode = 0
scores = []

EPISODES = 2000

"""
this code  copied and modified from below link
https://github.com/tensorflow/models/blob/master/research/a3c_blogpost/a3c_cartpole.py
"""


class Memory:
    def __init__(self, action_shape):
        self.action_shape = action_shape
        self.states = []
        self.rewards = []
        self.actions = []

    def reset(self):
        self.states = []
        self.rewards = []
        self.actions = []

    def store(self, state, action, reward):
        self.states.append(state)
        # to onehot
        act = np.zeros(self.action_shape)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)


class Actor:
    def __init__(self, state_shape, action_shape, lr, sess=None, name='actor'):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.lr = lr
        self.sess = sess
        self.name = name

        self.action_flatten_size = int(np.prod(self.action_shape))

        self.batch_state_shape = [None] + list(self.state_shape)
        self.batch_action_shape = [None] + list(self.action_shape)

        self.build()

    def build(self):
        with tf.variable_scope(self.name):
            self.states = placeholder(tf.float32, self.batch_state_shape, 'state')
            self.actions = placeholder(tf.float32, self.batch_action_shape, 'action')
            self.advantages = tf.placeholder(tf.float32, [None, ], 'advantages')

            stack = Stacker(self.states)
            stack.linear(64)
            stack.relu()
            stack.linear(64)
            stack.relu()
            stack.linear(self.action_flatten_size)
            stack.softmax()
            self.policy = stack.last_layer

            good_prob = tf.reduce_sum(self.actions * self.policy, axis=1)
            eligibility = tf.log(good_prob + 1e-10) * tf.stop_gradient(self.advantages)
            loss = -tf.reduce_sum(eligibility)
            entropy = tf.reduce_sum(self.policy * tf.log(self.policy + 1e-10), axis=1)
            self.loss = loss + 0.01 * entropy

            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, states, actions, advantages):
        feed_dict = {self.states: states, self.actions: actions, self.advantages: advantages}
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
        return loss

    def get_action(self, states):
        # todo shape... error
        states = np.reshape(states, [1] + list(self.state_shape))
        policys = self.sess.run(self.policy, {self.states: states})

        actions = []
        for policy in policys:
            actions += [np.random.choice(np.arange(self.action_flatten_size), p=policy)]

        # todo indice index out....
        return actions[0]

    # todo Implement
    def save(self, name):
        pass

    # todo Implement
    def load(self, name):
        pass


class Critic:
    def __init__(self, state_shape, lr, sess=None, name='critic'):
        self.state_shape = state_shape
        self.lr = lr
        self.sess = sess
        self.name = name
        self.batch_state_shape = [None] + list(self.state_shape)
        self.build()

    def build(self):
        with tf.variable_scope(self.name):
            self.states = placeholder(tf.float32, self.batch_state_shape, 'state')

            self.discounted_rewards = tf.placeholder(tf.float32, None, 'discounted_reward')

            stack = Stacker(self.states)
            stack.linear(64)
            stack.relu()
            stack.linear(64)
            stack.relu()
            stack.linear(1)
            self.value = stack.last_layer

            self.loss = tf.reduce_mean(tf.square(self.discounted_rewards - self.value))

            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, states, discounted_rewards):
        _, loss = self.sess.run(
            [self.train_op, self.loss],
            feed_dict={self.states: states, self.discounted_rewards: discounted_rewards}
        )
        return loss

    def predict(self, states):
        return self.sess.run(self.value, {self.states: states})

    # todo Implement
    def save(self, name):
        pass

    # todo Implement
    def load(self, name):
        pass


class A3CGlobal:
    def __init__(
            self,
            state_shape,
            action_shape,
            env_name,
            actor_lr=0.001,
            critic_lr=0.001,
            discount_factor=.99,
            sess=None
    ):
        self.state_shape = state_shape
        self.action_shape = action_shape

        self.env_name = env_name

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor

        self.sess = sess
        if self.sess is None:
            self.open()

        self.actor = Actor(self.state_shape, self.action_shape, self.actor_lr, sess=self.sess)
        self.critic = Critic(self.state_shape, self.critic_lr, sess=self.sess)

        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)

    def save(self, name):
        self.actor.save(name)
        self.critic.save(name)

    def load(self, name):
        self.actor.load(name)
        self.critic.load(name)

    def open(self):
        self.sess = tf.Session()

    def close(self):
        if self.sess is not None:
            self.sess.close()
            self.sess = None


class A3CLocal(threading.Thread):
    def __init__(self, index, env_name, a3c):
        threading.Thread.__init__(self)

        self.index = index
        self.env_name = env_name
        self.a3c = a3c
        self.actor = a3c.actor
        self.critic = a3c.critic
        self.discount_factor = a3c.discount_factor

        self.state_shape = a3c.state_shape

        self.memory = Memory(a3c.action_shape)

    def run(self):
        global episode

        env = gym.make(self.env_name)
        env._max_episode_steps = 1000
        while episode < EPISODES:
            state = env.reset()
            score = 0
            while True:
                action = self.actor.get_action(state)
                next_state, reward, done, _ = env.step(action)
                score += reward

                self.memory.store(state, action, reward)

                state = next_state

                if done:
                    break

            episode += 1
            print("episode: ", episode, "/ score : ", score)
            scores.append(score)
            self.train_episode(score != 500)

        print(f"thread {self.index} end")

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards, done=True):
        states = self.memory.states

        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        if not done:
            running_add = self.critic.predict(np.reshape(states[-1], (1, self.state_shape)))[0]

        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # update policy network and value network every episode
    def train_episode(self, done):
        rewards = self.memory.rewards
        states = self.memory.states
        actions = self.memory.actions

        discounted_rewards = self.discount_rewards(rewards, done)

        values = self.critic.predict(np.array(states))
        values = np.reshape(values, len(values))

        advantages = discounted_rewards - values

        self.actor.train(states, actions, advantages)
        self.critic.train(states, discounted_rewards)

        self.memory.reset()


class A3C:
    def __init__(
            self,
            state_shape,
            action_shape,
            env_name,
            actor_lr=0.001,
            critic_lr=0.001,
            discount_factor=.99,
            threads=8
    ):
        self.threads = threads
        self.env_name = env_name
        self.A3C_global = A3CGlobal(state_shape, action_shape, actor_lr, critic_lr, discount_factor)

    def train(self, episode):
        # self.load('./save_model/cartpole_a3c.h5')
        agents = [
            A3CLocal(i, self.env_name, self.A3C_global)
            for i in range(self.threads)
        ]

        for agent in agents:
            agent.start()

        while True:
            time.sleep(20)

            if episode >= EPISODES:
                break

        for agent in agents:
            agent.join()
        print(f'all agent joined')

    def save(self, name):
        self.A3C_global.save(name)

    def load(self, name):
        self.A3C_global.load(name)


def A3C_cartpole():
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    env.close()

    a3c = A3C(state_size, action_size, env_name, threads=8)
    a3c.train(2000)


def test_input_shape():
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    action_space_shape = np.array(env.action_space)
    state_space_shape = np.array(env.observation_space)

    action_space_shape = [2, ]
    state_space_shape = [4, ]

    # print(action_space_shape)
    # print(state_space_shape)
    env.close()

    a3c = A3C(state_space_shape, action_space_shape, env_name, threads=1)
    a3c.train(2000)

    pass
