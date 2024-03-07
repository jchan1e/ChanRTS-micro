#
# Using Gym, start a Game and attach an Agent to play it

import sys
import numpy as np
import gym
import gym_microrts
from gym_microrts import microrts_ai    # built-in AI agents
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv    # The Gym Env that runs the game
from agent import Agent
import csv

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


def loadAgents(env, list_of_filenames):
    agents = []
    for filename in list_of_filenames:
        A = Agent.fromFile(filename, "", env)
    return agents

if __name__ == "__main__":

    env0 = MicroRTSGridModeVecEnv(
            num_selfplay_envs=2,
            num_bot_envs=0,
            #ai2s=[bot_agent],
            #map_paths=["maps/4x4/base4x4.xml"]
            #map_paths=["maps/10x10/basesWorkers10x10.xml"]
            map_paths=["maps/16x16/basesWorkers16x16.xml"]
    )

    agents = loadAgents(env, sys.argv[1:])
    bots =  [   microrts_ai.randomBiasedAI,
                microrts_ai.workerRushAI,
                microrts_ai.lightRushAI,
                microrts_ai.mixedBot,
                microrts_ai.coacAI,
                microrts_ai.nativeMCTSAI
            ]
    matchups = [(a, b) for a in agents for b in bots]

    # game data format: { obs , action , reward , won_overall }
    game_data = []
    outfile = open(outfilename, "wb")
    writer = csv.DictWriter(outfile)

    for agent1, bot_agent in matchups:
        #bot_agent = microrts_ai.coacAI
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
        #agent1 = Agent.new("", env)
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
        #   # action type:                  [None, Move, Harvest, Return, Build, Attack]
        #   # move parameter:               [N, E, S, W]
        #   # harvest parameter:            [N, E, S, W]
        #   # return parameter:             [N, E, S, W]
        #   # produce_direction parameter:  [N, E, S, W]
        #   # produce_unit_type parameter:  [Resource, Base, Barrack, Worker, Light, Heavy, Ranged]
        #   # attack_target parameter
        #   sample(action_mask[:, 29 : sum(env.action_space.nvec[1:])]),
        #   ...
        #   Relative Attack Position    [0:a^2-1] Where a=7 (max attack range 3)

        # setting up data collection
        obs_seq = []
        action_seq = []
        reward_seq = []
        won = 0

        done = np.array([False])
        #nvec = env.action_space.nvec
        while not done.any():
            env.render()

            #time.sleep(1.0/30)

            #action = agent1.get_action(env, obs)
            action_mask = env.get_action_mask()
            action_mask = action_mask.reshape(-1, action_mask.shape[-1])
            #print(action_mask.shape)

            # inform the agent of the game state
            obs_seq.append(obs[0])
            agent1.set_obs(obs[0])
            agent1.set_action_mask(action_mask)

            # Get the agent's action vector
            action = agent1.get_action()
            action_seq.append(action)

            # Grid mode, concatenate all actions in an array
            obs, reward, done, info = env.step(action)
            reward_seq.append(reward)
            #winner = reward[0]
            if done.any():
                if reward >= 10.0:
                    won = 1
                elif reward < 0.0:
                    won = -1
                else:
                    won = 0

            #print(obs.shape)

        env.close()

        # Record obs and action sequences to game_data
        for i in xrange(len(action_seq)):
            game_data.append({"obs":obs_seq[i], "action":action_seq[i], "reward":reward_seq[i], "won":won})
            writer.write({"obs":obs_seq[i], "action":action_seq[i], "reward":reward_seq[i], "won":won})

        #agent1.save("test.agent")
