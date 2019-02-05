import numpy as np

from script.util.Stacker import Stacker
from script.util.tensor_ops import *
from script.workbench.gym_wrapper import gym_cartpole_v1_wrapper


class Actor:
    def __init__(self, state_size, action_size, lr=0.01, sess=None, name='actor'):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.sess = sess
        self.name = name

        self.build()

    def build(self):
        with tf.variable_scope(self.name):
            self.states = placeholder(tf.float32, [1, self.state_size], 'state')
            self.actions = tf.placeholder(tf.int32, None, 'action')
            self.td_error = tf.placeholder(tf.float32, None, 'td_error')

            stack = Stacker(self.states)
            stack.linear(64)
            stack.relu()
            stack.linear(64)
            stack.relu()
            stack.linear(self.action_size)
            stack.softmax()

            self.proba = stack.last_layer
            log_prob = tf.log(tf.clip_by_value(self.proba[0, self.actions], 1e-10, 1.0))

            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss
            # minimize(-exp_v) = maximize(exp_v)
            self.loss = - self.exp_v
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, states, actions, td_errors):
        feed_dict = {self.states: states, self.actions: actions, self.td_error: td_errors}
        _, exp_v = self.sess.run([self.train_op, self.loss], feed_dict)
        return exp_v

    def choose_action(self, states):
        probs = self.sess.run(self.proba, {self.states: states})  # get probabilities for all actions
        return np.random.choice(np.arange(len(probs[0])), p=probs[0])  # return a int


class Critic:
    def __init__(self, state_size, gamma=0.9, lr=0.01, sess=None, name='critic'):
        self.state_size = state_size
        self.gamma = gamma
        self.lr = lr
        self.sess = sess
        self.name = name

        self.build()

    def build(self):
        with tf.variable_scope(self.name):
            self.states = placeholder(tf.float32, [-1, self.state_size], "state")
            self.value_next = tf.placeholder(tf.float32, [1, 1], "v_next")
            self.rewards = tf.placeholder(tf.float32, None, 'rewards')

            stack = Stacker(self.states)
            stack.linear(64)
            stack.relu()
            stack.linear(64)
            stack.relu()
            stack.linear(1)
            self.value = stack.last_layer

            self.td_error = self.rewards + self.gamma * self.value_next - self.value
            self.loss = tf.square(self.td_error)

            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, states, rewards, next_states):
        v_next = self.predict(next_states)
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    feed_dict={self.states: states, self.value_next: v_next, self.rewards: rewards})
        return td_error

    def predict(self, state):
        return self.sess.run(self.value, {self.states: state})


class ActorCritic:
    def __init__(self, env, discount_factor, state_space, action_space, sess=None,
                 actor_lr=0.001, critic_lr=0.01,
                 weight_copy_interval=10, train_interval=10,
                 batch_size=32):
        self.env = env
        self.discount_factor = discount_factor
        self.state_space = state_space
        self.action_space = action_space

        self.sess = sess
        if self.sess is None:
            self.open()

        self.weight_copy_interval = weight_copy_interval
        self.batch_size = batch_size
        self.train_interval = train_interval

        self.actor = Actor(self.state_space, self.action_space, sess=self.sess, lr=actor_lr)
        self.critic = Critic(self.state_space, sess=self.sess, lr=critic_lr)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def open(self):
        self.sess = tf.Session()

    def close(self):
        if self.sess is not None:
            self.sess.close()
            self.sess = None

    def train(self, episodes):
        ma_reward = 0.0
        for i_episode in range(episodes):
            state = self.env.reset()
            while True:
                action = self.actor.choose_action(state)
                next_state, reward, done, info = self.env.step(action)

                if done: reward = -200

                td_error = self.critic.train(state, reward, next_state)
                # gradient = grad[reward + gamma * V(next_state) - V(state)]
                # true_gradient = grad[logPi(state,action) * td_error]
                actor_loss = self.actor.train(state, action, td_error)

                state = next_state

                if done:
                    break

            score = self.env.score
            ma_reward = ma_reward * 0.95 + score * 0.05
            print(f'episode = {i_episode}, reward={score}, ma={ma_reward:5.2f}')


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def ActorCritic_cartpole():
    env = gym_cartpole_v1_wrapper()
    # env.env._max_episode_steps = 10000

    observation_space = 4
    action_space = 2

    lr = 0.01
    discount_factor = 0.99
    max_episodes = 1000
    # max_episodes = 30

    train_interval = 10
    batch_size = 32
    actor_critic = ActorCritic(
        env, lr, discount_factor, observation_space, action_space,
        train_interval=train_interval,
        batch_size=batch_size,
        actor_lr=0.001,
        critic_lr=0.01,
    )
    actor_critic.train(max_episodes)
