import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
from torch.nn import functional as F


def plot_scatter(*args, **kwargs):
    plt.plot(*args, **kwargs)
    plt.scatter(*args, **kwargs)


def data_generator(x, y, batch_size):
    while True:
        for xy_pair in split((x, y), batch_size):
            yield xy_pair


def split(arr, size):
    arrays = []
    while len(arr) > size:
        slice_ = arr[:size]
        arrays.append(slice_)
        arr = arr[size:]
    arrays.append(arr)
    return arrays


def batcher(dataset, batch_size, infinite=False):
    while True:
        x, y = dataset
        for x_, y_ in zip(split(x, batch_size), split(y, batch_size)):
            yield x_, y_
        if not infinite:
            break


def load(checkpoint_name, model, optimiser):
    if os.path.exists(checkpoint_name):
        checkpoint = torch.load(checkpoint_name, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        grad_step = checkpoint['grad_step']
        #print(f'Restored checkpoint from {checkpoint_name}.')
        return grad_step
    return 0


def save(checkpoint_name, model, optimiser, grad_step):
    torch.save({
        'grad_step': grad_step,
        'model_state_dict': model.state_dict(),
        'optimiser_state_dict': optimiser.state_dict()
    }, checkpoint_name)


def train_100_grad_steps(checkpoint_name, data, device, net, optimiser, test_losses):
    global_step = load(checkpoint_name, net, optimiser)
    for x_train_batch, y_train_batch in data:
        global_step += 1
        optimiser.zero_grad()
        net.train()
        _, forecast = net(torch.tensor(x_train_batch, dtype=torch.float).to(device))
        loss = F.mse_loss(forecast, torch.tensor(y_train_batch, dtype=torch.float).to(device))
        loss.backward()
        optimiser.step()
        #Juan
        #if global_step % 30 == 0:
            #print(f'grad_step = {str(global_step).zfill(6)}, tr_loss = {loss.item():.6f}, te_loss = {test_losses[-1]:.6f}')
        if global_step > 0 and global_step % 100 == 0:
            with torch.no_grad():
                save(checkpoint_name, net, optimiser, global_step)
            break


def fit(checkpoint_name, net, optimiser, data_generator, on_save_callback, device, max_grad_steps=10000):
    #print('--- Training ---')
    initial_grad_step = load(checkpoint_name, net, optimiser)
    for grad_step, (x, target) in enumerate(data_generator):
        grad_step += initial_grad_step
        optimiser.zero_grad()
        net.train()
        backcast, forecast = net(torch.tensor(x, dtype=torch.float).to(device))
        loss = F.mse_loss(forecast, torch.tensor(target, dtype=torch.float).to(device))
        loss.backward()
        optimiser.step()
        #print(f'grad_step = {str(grad_step).zfill(6)}, loss = {loss.item():.6f}')
        if grad_step % 1000 == 0 or (grad_step < 1000 and grad_step % 100 == 0):
            with torch.no_grad():
                save(checkpoint_name, net, optimiser, grad_step)
                if on_save_callback is not None:
                    on_save_callback(x, target, grad_step)
        if grad_step > max_grad_steps:
            print('Finished.')
            break


def eval_test(backcast_length, forecast_length, net, norm_constant, test_losses, x_test, y_test):
    net.eval()
    _, forecast = net(torch.tensor(x_test, dtype=torch.float))
    singular_loss = F.mse_loss(forecast, torch.tensor(y_test, dtype=torch.float)).item()
    test_losses.append(singular_loss)
    #Juan
    #p = forecast.detach().numpy()
    
    p = forecast.detach().cpu().numpy()
    '''
    subplots = [221, 222, 223, 224]
    plt.figure(1)
    for plot_id, i in enumerate(np.random.choice(range(len(x_test)), size=4, replace=False)):
        ff, xx, yy = p[i] * norm_constant, x_test[i] * norm_constant, y_test[i] * norm_constant
        plt.subplot(subplots[plot_id])
        plt.grid()
        plot_scatter(range(0, backcast_length), xx, color='b')
        plot_scatter(range(backcast_length, backcast_length + forecast_length), yy, color='g')
        plot_scatter(range(backcast_length, backcast_length + forecast_length), ff, color='r')
    plt.show()
    
    '''


def get_avg_score(net, x_test, y_test):
    net.eval()
    _, forecast = net(torch.tensor(x_test, dtype=torch.float))
    singular_loss = F.mse_loss(forecast, torch.tensor(y_test, dtype=torch.float)).item()
    return singular_loss


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
    #print(norm_constant)
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
    x_train, x_test, y_train, y_test = train_test_split(x_train_batch, y, test_size=0.005, random_state=17)
    #x_train, y_train = x_train_batch[:c], y[:c]
    #x_test, y_test = x_train_batch[c:], y[c:]
    #print(x_train.shape, x_test.shape)
    #print(y_train.shape, y_test.shape)
    data = data_generator(x_train, y_train, batch_size)

    return data, x_test, y_test, norm_constant


def organise_data(data, data_header, forecast_length, backcast_length, batch_size):
    normal_signal_data = []
    normal_signal_x = []

    normal_signal_data.append(data[3])
    normal_signal_x.append(range(0, int(data_header[0].split(' ')[3])))

    normal_signal_data = [y for sublist in normal_signal_data for y in sublist]
    normal_signal_x = [y for sublist in normal_signal_x for y in sublist]
    normal_signal_data = np.array(normal_signal_data)
    normal_signal_x = np.array(normal_signal_x)
    normal_signal_data.flatten()
    normal_signal_x.flatten()

    norm_constant = np.max(normal_signal_data)
    #print(norm_constant)
    normal_signal_data = normal_signal_data / norm_constant  # leak to the test set here.

    x, y = [], []
    for i in range(backcast_length, len(normal_signal_data) - forecast_length):
        x.append(normal_signal_data[i - backcast_length:i])
        y.append(normal_signal_data[i:i + forecast_length])

    x = np.array(x)  # [..., 0]
    y = np.array(y)  # [..., 0]

    if len(x) > 5000:
        x = x[0:5000]
        y = y[0:5000]

    return x, y
