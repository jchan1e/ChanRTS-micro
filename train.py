
import sys
import csv
import random
import agent # includes torch, torch.nn, torch.optim
from agent import Agent
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
import torch
import torch.nn as nn
import torch.optim as optim


if __name__ == "__main__":
    datafiles = sys.argv[2:]
    agentfile = sys.argv[1]

    A = Agent.fromFile(agentfile, "")
    data = []
    #data = [{}]
    for filename in datafiles:
        full_data = []
        with open(filename, "rb") as F:
            reader = csv.DictReader(F)
            for line in reader:
                full_data.append(line)
        # train on 1/10 of the frames from each game
        data = data + random.sample(full_data, len(full_data)/10)
    data = np.array(data)
    np.random.shuffle(data)

    train_size = len(data)*0.8
    test_size = len(data)-train_size

    train_set, test_set = torch.utils.data.random_split(data, [train_size, test_size])

    learning_rate = 2.5e-4
    eps = 1e-5

    optimizer = optim.Adam(A.model.parameters(), lr=learning_rate, eps=eps)
    loss_fn = nn.MSELoss()

    for x_train, y_train in training_set:

        v_pred = A.model.get_value(x_train)
        v_loss = loss_fn(v_pred, v_train)

        y_pred, logprob, entropy, invalid_action_mask = A.get_action(x_train)
        e_loss = 0.0
        pg_loss = 0.0
        loss = pg_logg - e_coef*e_loss + v_coef*v_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    A.save(agentfile)
