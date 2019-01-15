import numpy as np
import gym
import tqdm
import tensorflow as tf


#
# def e_greedy(actions, Q, state, e=0.1, decay=0.9):
#     # return action
#     e = e * decay
#
#     if random.uniform(0, 1) < e:
#         action = random.choice(actions)
#     else:
#         action = argmax(Q(s, a))
#
#     return action

def add_noise():
    # nn 을 사용할거라면 벡터에 노이즈를 넣을수도있음
    # e_greedy 보다는 덜 랜덤함
    # 기존 q 값을 기반으로 함
    pass


class Q_Learning:
    def __init__(self, lr, discount_factor, state_space, action_space, decay=1.):
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


def q_learning_frozenlake():
    env = gym.make('FrozenLake-v0', )

    lr = .85
    discount_factor = .99
    num_episodes = 2000
    q_learning = Q_Learning(lr, discount_factor, env.observation_space.n, env.action_space.n)

    rList = []
    for episode in tqdm.trange(num_episodes):
        state = env.reset()

        reward_sum = 0
        done = False
        for move_count in range(99):
            action = q_learning.chose(state, episode)

            # env.step은 주어진 행동에 대한 다음 상태와 보상, 끝났는지 여부, 추가정보를 제공함
            next_state, reward, done, _ = env.step(action)

            q_learning.update(state, action, next_state, reward)

            reward_sum += reward

            state = next_state

            if done:
                break

        rList.append(reward_sum)

    print("Score over time: " + str(sum(rList) / num_episodes))


class QNetwork:
    def __init__(self, lr, discount_factor, state_space, action_space, decay=1, sess=None):
        self.lr = lr
        self.discount_factor = discount_factor
        self.state_space = state_space
        self.action_space = action_space

        self.sess = sess
        if self.sess is None:
            self.sess = tf.Session()
        self.e = 1

        self.build_network()

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

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
        self.e = 1. / ((episode / 50) + 10)

    def Q_S(self, state):
        allQ = self.sess.run(self.Qout, feed_dict={self.inputs1: np.identity(16)[state:state + 1]})

        return allQ

    def chose(self, state, env):
        a = self.sess.run(self.predict, feed_dict={self.inputs1: np.identity(16)[state:state + 1]})

        if np.random.rand(1) < self.e:
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

def qnetwork_frozenlake():
    env = gym.make('FrozenLake-v0', )

    lr = .85
    discount_factor = .99
    num_episodes = 2000

    tf.reset_default_graph()
    with tf.Session() as sess:
        qnetwork = QNetwork(lr, discount_factor, env.observation_space.n, env.action_space.n, sess=sess)
        rList = []
        for episode in tqdm.trange(num_episodes):
            state = env.reset()

            reward_sum = 0
            done = False

            for move_count in range(99):
                action = qnetwork.chose(state, env)

                # env.step은 주어진 행동에 대한 다음 상태와 보상, 끝났는지 여부, 추가정보를 제공함
                next_state, reward, done, _ = env.step(action)

                qnetwork.update(state, action, next_state, reward)

                reward_sum += reward

                state = next_state

                if done:
                    print(move_count, reward_sum)
                    break

            qnetwork.decay_e(episode)

            rList.append(reward_sum)

        print("Score over time: " + str(sum(rList) / num_episodes))


# def q_table_algorithm():
#     env = gym.make('FrozenLake-v0')
#
#
#     def policy(state):
#         # return action by policy
#         return 0
#
#     def update_q_function(old_state, new_state, reward, action, lr, discount_factor):
#         q = {}
#         q[(old_state, action)] += lr * (
#                 reward + discount_factor * argmax(q[(new_state, new_action)]) - q[(old_state, action)])
#
#         return
#
#     env = None
#
#     lr = 0.8
#     discount_factor = 0.3
#     nam_episode = 1000
#     for i in range(nam_episode):
#         env.reset()
#
#         # get state from env
#         state = env.get_init_state()
#
#         reward_sum = 0.
#         while env.is_end():
#             # get action from s by policy function
#             action = policy(state)
#
#             # do action and get reward and new state
#             reward, new_state = env.do_action(action)
#
#             # update q function
#             # q(s,a) = r + q(s', a')
#             update_q_function(state, new_state, reward, action, lr, discount_factor)
#
#             # assign new state
#             state = new_state
#             reward_sum += reward
#
#         print(f'reward sum= {reward_sum}')


def Q_nn_learning():
    pass
