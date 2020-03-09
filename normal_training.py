from nbeats_pytorch.model import NBeatsNet
import nbeats_additional_functions as naf
import pandas as pd
import numpy as np
import wfdb
import os
import torch
from torch.nn import functional as F
from torch import optim
import matplotlib.pyplot as plt


def one_file_training_data(data_dir, file, forecast_length, backcast_length, batch_size):
    normal_signal_data = []
    normal_signal_x = []

    x = wfdb.io.rdsamp(data_dir + file[:-4])
    normal_signal_data.append(x[0][:, 3])
    normal_signal_x.append(range(0, int(x[1]['sig_len'])))

    normal_signal_data = [y for sublist in normal_signal_data for y in sublist]
    normal_signal_x = [y for sublist in normal_signal_x for y in sublist]
    normal_signal_data = np.array(normal_signal_data)
    normal_signal_x = np.array(normal_signal_x)
    normal_signal_data.flatten()
    normal_signal_x.flatten()

    norm_constant = np.max(normal_signal_data)
    print(norm_constant)
    normal_signal_data = normal_signal_data / norm_constant  # leak to the test set here.

    x_train_batch, y = [], []
    for i in range(backcast_length, len(normal_signal_data) - forecast_length):
        x_train_batch.append(normal_signal_data[i - backcast_length:i])
        y.append(normal_signal_data[i:i + forecast_length])

    x_train_batch = np.array(x_train_batch)  # [..., 0]
    y = np.array(y)  # [..., 0]

    if len(x_train_batch) > 30000:
        x_train_batch = x_train_batch[0:int(len(x_train_batch) / 4)]
        y = y[0:int(len(y) / 4)]

    c = int(len(x_train_batch) * 0.8)
    x_train, y_train = x_train_batch[:c], y[:c]
    x_test, y_test = x_train_batch[c:], y[c:]
    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)
    data = naf.data_generator(x_train, y_train, batch_size)

    return data, x_test, y_test, norm_constant


checkpoint_name_BASE = "nbeats_checkpoint.th"

data_dir = os.getcwd() + "/data/"

d = [x[0] for x in os.walk(data_dir)]
dirs = []
for directory_name in d:
    if ',' not in directory_name:
        dirs.append(directory_name)
dirs = dirs[1:]
print(dirs)

device = torch.device('cpu')
forecast_length = 500
backcast_length = 3 * forecast_length
batch_size = 256

for folder_name in dirs:
    name = folder_name.split("/")[-1]
    checkpoint_name = name + "_" + checkpoint_name_BASE

    if os.path.isfile(checkpoint_name):
        continue

    net = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
                    forecast_length=forecast_length,
                    thetas_dims=[7, 8],
                    nb_blocks_per_stack=3,
                    backcast_length=backcast_length,
                    hidden_layer_units=128,
                    share_weights_in_stack=False,
                    device=device)
    optimiser = optim.Adam(net.parameters())

    test_losses = []
    actual_class_dir = data_dir + "/" + name + "/"
    for (_, dirs, files) in os.walk(actual_class_dir):
        iteration = 0
        for file in files:

            if 'mat' in file:
                continue

            iteration += 1
            print(iteration)
            if iteration > 8:
                break

            data, x_test, y_test, norm_constant = one_file_training_data(actual_class_dir, file, forecast_length,
                                                                         backcast_length, batch_size)

            for i in range(10):
                naf.eval_test(backcast_length, forecast_length, net, norm_constant, test_losses, x_test, y_test)
                naf.train_100_grad_steps(checkpoint_name, data, device, net, optimiser, test_losses)
