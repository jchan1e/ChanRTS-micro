#
# generic agent class which can be trained and can play in real time

import numpy as np
import gym_microrts
import torch
import torch.nn as nn
import torch.optim as optim
#from torch.distribution.categorical import Categorical

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def softmax(x, axis=None):
    x = x-x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y/y.sum(axis=axis, keepdims=True)

def sample(x):
    p = softmax(x, axis=1)
    #print(p.shape)
    c = p.cumsum(axis=1)
    u = np.random.rand(len(c), 1)
    choices = (u < c).argmax(axis=1)
    #print(choices.shape)
    return choices.reshape(-1,1)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Model(nn.Module):
    #mapsize = None
    def __init__(self, obs_shape, action_shape):
        super().__init__()
        mapsize = (obs_shape[0], obs_shape[1])
        self.network = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[2], 20, kernel_size=3, padding=1)),
                nn.ReLU(),
                layer_init(nn.Conv2d(20, 32, kernel_size=3, padding=1)),
                nn.ReLU(),
                nn.Flatten(0,-1),
                layer_init(nn.Linear(32*mapsize[0]*mapsize[1], 256)),
                #layer_init(nn.Linear(32*16*16, 256)),
                nn.ReLU(),
                )
        self.actor = layer_init(nn.Linear(256, np.prod(action_shape)), std=0.01)

    def forward(self, x):
        #print(x.shape)
        y = self.network(x.permute((2,0,1)))
        #print(y.shape)
        return self.actor(y)

class Agent():
    # class variables
    env = None
    obs = None
    action = None
    model = None
    mapsize = None

    def __init__(self, agent_type, env):
        # assign class variables
        self.env = env
        # populate initial observation

        # build model
        #if agent_type == "simple":
        #    pass
        obs_shape = env.observation_space.shape
        action_shape = env.get_action_mask().shape
        #print(obs_shape)
        #print(action_shape)
        #self.model = Model(obs_shape, action_shape)
        #self.model = Model(obs_shape, action_shape).to(device)

    @classmethod
    def new(cls, agent_type, env):
        A = cls(agent_type, env)
        obs_shape = env.observation_space.shape
        action_shape = env.get_action_mask().shape
        A.model = Model(obs_shape, action_shape).to(device)
        return A

    @classmethod
    def fromFile(cls, modelfile, agent_type, env):
        A = cls(agent_type, env)
        A.model = torch.load(modelfile).to(device)
        return A

    def save(self, filename):
        #f = open(filename, "w")
        torch.save(self.model, filename)
        #f.close()

    def set_obs(self, obs):
        self.obs = torch.from_numpy(obs.astype(np.float32)).to(device)

    def set_action_mask(self, action_mask):
        self.action_mask = torch.from_numpy(action_mask).to(device)

    def get_action(self):
        y = self.model.forward(self.obs)
        #print(self.action_mask.shape)
        #print(y.shape)
        y = y.reshape(self.action_mask.shape)
        y = self.action_mask * y
        y = y.cpu().detach().numpy() # This is where it should return to main RAM and run on cpu
        self.action = np.concatenate(
            (
                sample(y[:, 0:6]),
                sample(y[:, 6:10]),
                sample(y[:, 10:14]),
                sample(y[:, 14:18]),
                sample(y[:, 18:22]),
                sample(y[:, 22:29]),
                sample(y[:, 29:78]),
            ),
            axis=1,
        )
        #print(self.action.shape)
        return self.action
