#
# Using Gym, start a Game and attach an Agent to play it

import sys
import numpy as np
import gym
import gym_microrts
from gym_microrts import microrts_ai    # built-in AI agents
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv    # The Gym Env that runs the game
from agent import Agent

import time

np.set_printoptions(threshold=sys.maxsize)


#def softmax(x, axis=None):
#    x = x - x.max(axis=axis, keepdims=True)
#    y = np.exp(x)
#    return y / y.sum(axis=axis, keepdims=True)
#
#def sample(logits):
#    # https://stackoverflow.com/a/40475357/6611317
#    # Takes an array of weighted probabilities, normalizes using softmax,
#    #   selects one by its probability, then returns the array of indeces chosen
#    p = softmax(logits, axis=1)
#    c = p.cumsum(axis=1)
#    u = np.random.rand(len(c), 1)
#    choices = (u < c).argmax(axis=1)
#    return choices.reshape(-1, 1)


if __name__ == "__main__":
    bot_agent = microrts_ai.coacAI
    env = MicroRTSGridModeVecEnv(
            num_selfplay_envs=0,
            num_bot_envs=1,
            ai2s=[bot_agent],
            #map_paths=["maps/4x4/base4x4.xml"]
            #map_paths=["maps/10x10/basesWorkers10x10.xml"]
            map_paths=["maps/16x16/basesWorkers16x16.xml"],
            reward_weight=np.array([10.0,1.0,1.0,0.2,1.0,4.0])
    )
    obs = env.reset()
    agent1 = Agent.new("", env)
    #agent1 = Agent.fromFile("test.agent", "", env)

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

    #print("Action Space: ", env.action_space)
    done = np.array([False,False,False])
    nvec = env.action_space.nvec
    while not done.any():
        env.render()

        #time.sleep(1.0/30)

        #action = agent1.get_action(env, obs)
        action_mask = env.get_action_mask()
        action_mask = action_mask.reshape(-1, action_mask.shape[-1])
        #print(action_mask.shape)
        # Set invalid actions to strong negative value
        #action_mask[action_mask == 0] = -9e8

        # Combine action mask with Agent Output
        #action_mask[action_mask == 1] = agent1.get_action(obs)

        # randomly sample valid action parameters
        #action = np.concatenate(
        #    (
        #        sample(action_mask[:, 0:6]),    # action type:                  [None, Move, Harvest, Return, Build, Attack]
        #        sample(action_mask[:, 6:10]),   # move parameter:               [N, E, S, W]
        #        sample(action_mask[:, 10:14]),  # harvest parameter:            [N, E, S, W]
        #        sample(action_mask[:, 14:18]),  # return parameter:             [N, E, S, W]
        #        sample(action_mask[:, 18:22]),  # produce_direction parameter:  [N, E, S, W]
        #        sample(action_mask[:, 22:29]),  # produce_unit_type parameter:  [Resource, Base, Barrack, Worker, Light, Heavy, Ranged]
        #        # attack_target parameter
        #        sample(action_mask[:, 29 : sum(env.action_space.nvec[1:])]),
        #    ),
        #    axis=1,
        #)
        #print(obs)
        #print(action)
        #print()

        # inform the agent of the game state
        agent1.set_obs(obs[0])
        agent1.set_action_mask(action_mask)

        # Get the agent's action vector
        action, _, _, _ = agent1.get_action()

        # Grid mode, concatenate all actions in an array
        obs, reward, done, info = env.step(action.cpu().numpy())
        print(done, reward)
        #print(obs.shape)

    env.close()

    #agent1.save("test.agent")
