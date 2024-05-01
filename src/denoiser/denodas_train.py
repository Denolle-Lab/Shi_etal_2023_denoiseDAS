import h5py
import time
import torch
import configparser
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from numpy.random import default_rng
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from das_denoise_models import unet, dataflow


def train(configure_file='config.ini'):
    #data
    datah5 = '/fd1/QibinShi_data/akdas/decimator2_2023-06-18_09.15.03_UTC.h5'
    with h5py.File(datah5, 'r') as f:
        time_data = f['Acquisition']['Raw[0]']['RawData'][:1500, 100:3100]
        delta_space = f['Acquisition'].attrs['SpatialSamplingInterval']
        sample_rate = f['Acquisition']['Raw[0]'].attrs['OutputDataRate']

    x = time_data.T[np.newaxis, :, :]
    x = np.repeat(x, 5, axis=0)
    print(x.shape, x.dtype)
    training_data = dataflow(x[:4])
    validation_data = dataflow(x[4:])

    """ Initialize the U-net model """
    model = unet(1, 16, 1024, factors=(5, 3, 2, 2))
    devc = try_gpu(i=1)
    model.to(devc)
    # %% Hyper-parameters for training
    batch_size = 10
    lr = 1e-3
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    train_iter = DataLoader(training_data, batch_size=batch_size, shuffle=False)
    validate_iter = DataLoader(validation_data, batch_size=batch_size, shuffle=False)


    model, avg_train_losses, avg_valid_losses = train_augmentation(train_iter,
                                                                   validate_iter,
                                                                   model,
                                                                   loss_fn,
                                                                   optimizer,
                                                                   epochs=50,
                                                                   patience=6,
                                                                   device=devc,
                                                                   minimum_epochs=5)



def try_gpu(i=0):  # @save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): trace print function.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_augmentation(train_dataloader, validate_dataloader, model, loss_fn, optimizer, lr_schedule, epochs,
                        patience, device, minimum_epochs=None):
    # get early_stopping ready
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # save history of losses every epoch
    avg_train_losses = []
    avg_valid_losses = []

    for epoch in range(1, epochs + 1):
        starttime = time.time()  # record time for each epoch
        train_losses = []  # save loss for every batch
        valid_losses = []

        # ======================= training =======================
        model.train()  # train mode on
        for batch, ((X, mask), y) in enumerate(train_dataloader):
            X, mask, y = X.to(device), mask.to(device), y.to(device)

            # predict and loss
            pred = model(X * mask)
            loss = loss_fn(pred * (1-mask), y)
            train_losses.append(loss.item())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ======================= validating =======================
        model.eval()  # evaluation model on
        with torch.no_grad():
            for (X, mask), y in validate_dataloader:
                X, mask, y = X.to(device), mask.to(device), y.to(device)

                pred = model(X * mask)
                loss = loss_fn(pred * (1-mask), y)
                valid_losses.append(loss.item())

        # average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        lr_schedule.step(valid_loss)

        # ==================== history monitoring ====================
        # print training/validation statistics
        epoch_len = len(str(epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f} ' +
                     f'time per epoch: {(time.time() - starttime):.3f} s')
        print(print_msg)

        if (minimum_epochs is None) or ((minimum_epochs is not None) and (epoch > minimum_epochs)):
            # if the current valid loss is lowest, save a checkpoint of model weights
            early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint as the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    return model, avg_train_losses, avg_valid_losses

if __name__ == '__main__':
    train()