import numpy as np
import tqdm

from script.util.Stacker import Stacker
from script.util.tensor_ops import *
from script.workbench.gym_wrapper import gym_cartpole_v1_wrapper


class QNetwork:
    def __init__(self, input_size, output_size, lr, name='main', sess=None, dueling=True):
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.name = name
        self.sess = sess
        self.dueling = dueling

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


class PolicyNetwork:
    def __init__(self, lr, input_size, output_size, sess=None, name='policyNetwork'):
        self.lr = lr
        self.input_size = input_size
        self.output_size = output_size
        self.sess = sess
        self.name = name
        self.build_network()

    def build_network(self):
        with tf.variable_scope(self.name):
            self.x = placeholder(tf.float32, [-1, self.input_size], name='x')
            self.y = placeholder(tf.float32, [-1, self.output_size], 'Y')
            self.discounted_episode_reward = placeholder(tf.float32, [-1, ], 'discounted_episode_reward')
            stack = Stacker(self.x)
            stack.linear(64)
            stack.relu()
            stack.linear(64)
            stack.relu()
            stack.linear(self.output_size)
            stack.softmax()

            self._proba = stack.last_layer
            neg_log_proba = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self._proba, labels=self.y)

            self.loss = tf.reduce_mean(neg_log_proba * self.discounted_episode_reward, name='loss')

            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def proba(self, xs):
        return self.sess.run(self._proba, feed_dict={self.x: xs})

    def train(self, xs, ys, discounted_episode_reward):
        loss, _ = self.sess.run(
            [self.loss, self.train_op],
            feed_dict={
                self.x: xs,
                self.y: ys,
                self.discounted_episode_reward: discounted_episode_reward
            }
        )
        return loss


class PolicyGradient:
    def __init__(self, env, lr, discount_factor, state_size, action_state, name='main', sess=None):
        self.env = env
        self.state_size = state_size
        self.action_size = action_state
        self.lr = lr
        self.name = name
        self.sess = sess

        if self.sess is None:
            self.open()
        self.gamma = discount_factor
        self.gamma = 0.95

        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

        self.polict_network = PolicyNetwork(self.lr, self.state_size, self.action_size, self.sess)
        # self.build_network()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def open(self):
        self.sess = tf.Session()

    def close(self):
        if self.sess is not None:
            self.sess.close()
            self.sess = None

    def chose(self, state):
        # Choose action a, remember WE'RE NOT IN A DETERMINISTIC ENVIRONMENT, WE'RE OUTPUT PROBABILITIES.
        action_probability_distribution = self.polict_network.proba(state.reshape([1, 4]))

        action = np.random.choice(range(action_probability_distribution.shape[1]),
                                  p=action_probability_distribution.ravel())  # select action w.r.t the actions prob
        return action

    def discount_rewards(self, episode_rewards):
        discounted_episode_rewards = np.zeros_like(episode_rewards)
        cumulative = 0.0
        for i in reversed(range(len(episode_rewards))):
            cumulative = cumulative * self.gamma + episode_rewards[i]
            discounted_episode_rewards[i] = cumulative

        return discounted_episode_rewards

    def normalize_reward(self, rewards):
        mean = np.mean(rewards)
        std = np.std(rewards)
        rewards = (rewards - mean) / std

        return rewards

    def update(self):
        # Calculate discounted reward
        discounted_rewards = self.discount_rewards(self.episode_rewards)
        discounted_rewards = self.normalize_reward(discounted_rewards)

        # Feedforward, gradient and backpropagation
        loss = self.polict_network.train(
            np.vstack(np.array(self.episode_states)),
            np.vstack(np.array(self.episode_actions)),
            discounted_rewards
        )

        return loss

    def reset_history(self):
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

    def append_history(self, state, action, reward):
        # Store s, a, r
        self.episode_states.append(state)

        # For actions because we output only one (the index) we need 2 (1 is for the action taken)
        # We need [0., 1.] (if we take right) not just the index
        # transform to onehot
        action_ = np.zeros(self.action_size)
        action_[action] = 1
        self.episode_actions.append(action_)

        self.episode_rewards.append(reward)

        return

    def train(self, episodes):
        allRewards = []
        total_rewards = 0
        maximumRewardRecorded = 0

        for episode in tqdm.trange(episodes):
            self.reset_history()
            state = self.env.reset()

            while True:
                action = self.chose(state)

                # Perform action
                new_state, reward, done, info = self.env.step(action)

                self.append_history(state, action, reward)

                if done:
                    self.update()

                    # Calculate sum reward
                    episode_rewards_sum = np.sum(self.episode_rewards)
                    allRewards.append(episode_rewards_sum)

                    total_rewards = np.sum(allRewards)

                    # Mean reward
                    mean_reward = np.divide(total_rewards, episode + 1)

                    maximumRewardRecorded = np.amax(allRewards)

                    tqdm.tqdm.write(
                        f"e: {episode}, r = {episode_rewards_sum}, mean={mean_reward:.3}, max={maximumRewardRecorded}")

                    break

                state = new_state

        return allRewards

        # for episode in range(episodes):
        #     state = self.env.reset()
        #     reward_sum = 0.0
        #     while True:
        #
        #         action = np.
        #         next_state, reward, done, _ = self.env.step(action)
        #
        #         if done:
        #             break
        #
        # pass


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def policy_gradient_cartpole():
    env = gym_cartpole_v1_wrapper()

    observation_space = 4
    action_space = 2

    lr = 0.01
    discount_factor = 0.99
    max_episodes = 3000
    # max_episodes = 30
    pg = PolicyGradient(env, lr, discount_factor, observation_space, action_space)
    reward_sums = pg.train(max_episodes)

    reward_sums = moving_average(reward_sums, 100)
    print(reward_sums)
    print("Score over time: " + str(sum(reward_sums) / max_episodes))

    # from script.util.PlotTools import PlotTools
    # plot = PlotTools(save=False, show=True)
    # plot.plot(reward_sums)
