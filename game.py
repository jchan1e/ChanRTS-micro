#
# Using Gym, start a Game and attach an Agent to play it

import numpy as np
import gym
import gym_microrts
from gym_microrts import microrts_ai    # built-in AI agents
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv    # The Gym Env that runs the game


class Agent:
    def get_action(self, env, obs):
        return env.action_sample.sample()

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def sample(logits):
    # https://stackoverflow.com/a/40475357/6611317
    # Takes an array of weighted probabilities, normalizes using softmax,
    #   selects one by its probability, then returns the array in 1-hot form
    p = softmax(logits, axis=1)
    c = p.cumsum(axis=1)
    u = np.random.rand(len(c), 1)
    choices = (u < c).argmax(axis=1)
    return choices.reshape(-1, 1)


if __name__ == "__main__":
    agent1 = microrts_ai.coacAI
    agent2 = microrts_ai.mixedBot
    env = MicroRTSGridModeVecEnv(
            num_selfplay_envs=2,
            num_bot_envs=1,
            ai2s=[agent1]
    )

    # Observation space:
    #   Feature     #_planes    Values
    #
    #   HP          5           [0, 1, 2, 3, 4+]
    #   Resources   5           [0, 1, 2, 3, 4+]
    #   Owner       3           [P1, None, P2]
    #   Unit Types  8           [None, Resource, Base, Barrack, Worker, Light, Heavy, Ranged]
    #   Current Action  6       [None, Move, Harvest, Return, Build, Attack]

    # Action Space
    #   Source Unit     [0:h*w]
    #   ...
    #   (see below)
    #   ...
    #   Relative Attack Position    [0:a^2-1] Where a=7 (max attack range 3)

    done = np.array([False,False,False])
    obs = env.reset()[0]
    nvec = env.action_space.nvec
    while not done.any():
        env.render()

        #action = agent1.get_action(env, obs)
        action_mask = env.get_action_mask()
        action_mask = action_mask.reshape(-1, action_mask.shape[-1])
        # Set invalid actions to strong negative value
        action_mask[action_mask == 0] = -9e8

        # Combine action mask with Agent Output

        # sample valid actions
        action = np.concatenate(
            (
                sample(action_mask[:, 0:6]),    # action type:                  [None, Move, Harvest, Return, Build, Attack]
                sample(action_mask[:, 6:10]),   # move parameter:               [N, E, S, W]
                sample(action_mask[:, 10:14]),  # harvest parameter:            [N, E, S, W]
                sample(action_mask[:, 14:18]),  # return parameter:             [N, E, S, W]
                sample(action_mask[:, 18:22]),  # produce_direction parameter:  [N, E, S, W]
                sample(action_mask[:, 22:29]),  # produce_unit_type parameter:  [Resource, Base, Barrack, Worker, Light, Heavy, Ranged]
                # attack_target parameter
                sample(action_mask[:, 29 : sum(env.action_space.nvec[1:])]),
            ),
            axis=1,
        )
        obs, reward, done, info = env.step(action)

    env.close()
