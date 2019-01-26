import gym
import numpy as np


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
    def action_space_sample(self):
        return

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
