import torch
from torch import nn
import torch.nn.functional as F

import numpy as np


class unet(nn.Module):
    """ Stack convolutional blocks. """

    def __init__(self, ch_in, ch0, ch_max, factors=None, kernel=(3, 3), pad=(1, 1), use_att=False):
        """ must specify:
            ch_in:   channel number of data
            ch0:     channel number of first feature map
            ch_max:  maximum number of channels
            factors: resample factors of every block
        """
        super(unet, self).__init__()
        self.level = len(factors)  # number of blocks downwards/upwards
        self.factor = factors
        self.relu = nn.ReLU()
        self.kernel = kernel
        self.pad = pad
        self.use_att = use_att
        self.layer = nn.ModuleList([])
        if self.use_att:
            self.attgates = nn.ModuleList([])
            for i in range(self.level):
                nch = min(ch0 * 2 ** i, ch_max)
                self.attgates.append(AttentionGate(nch))

        """Deepening to extract features"""
        for i in range(self.level+1):
            # %% an unit level, conv+conv+pool
            if i == 0:
                nch_input = ch_in
            else:
                nch_input = nch_output
            nch_output = min(ch0 * 2**i, ch_max)
            self.layer.append(nn.Conv2d(nch_input, nch_output, self.kernel, padding=self.pad))
            self.layer.append(nn.Conv2d(nch_output, nch_output, self.kernel, padding=self.pad))

            if i > self.level-2:  # only drop out at the bottom two levels
                self.layer.append(nn.Dropout(p=0.2))
            if i < self.level:
                self.layer.append(MaxBlurPool2d(nch_output, kernel_size=(self.factor[i], self.factor[i])))

        """Shallowing to reconstruct wavefields"""
        for i in range(self.level):
            # %% an unit level, upsample+conv+conv+conv
            nch_input = min(ch0 * 2 ** (self.level - i), ch_max)
            nch_output = min(ch0 * 2 ** (self.level - i - 1), ch_max)
            scale_factor = (self.factor[self.level - 1 - i], self.factor[self.level - 1 - i])
            self.layer.append(nn.Upsample(scale_factor=scale_factor, mode='nearest'))
            self.layer.append(nn.Conv2d(nch_input, nch_output, self.kernel, padding=self.pad))
            # skip-connect will be here
            self.layer.append(nn.Conv2d(nch_input, nch_output, self.kernel, padding=self.pad))
            self.layer.append(nn.Conv2d(nch_output, nch_output, self.kernel, padding=self.pad))

        self.layer.append(nn.Conv2d(nch_output, ch_in, self.kernel, padding=self.pad))

        self.initialize_weights()

    def forward(self, x):
        cat_content = []
        """Deepening to extract features"""
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            
        for i in range(self.level-1):
            x = self.layer[3 * i + 0](x)  # conv1
            x = self.relu(x)
            x = self.layer[3 * i + 1](x)  # conv2
            x = self.relu(x)
            cat_content.append(x)
            x = self.layer[3 * i + 2](x)  # pool

        x = self.layer[3 * (self.level - 1) + 0](x)  # conv1
        x = self.relu(x)
        x = self.layer[3 * (self.level - 1) + 1](x)  # conv2
        x = self.relu(x)
        x = self.layer[3 * (self.level - 1) + 2](x)  # dropout
        cat_content.append(x)
        x = self.layer[3 * (self.level - 1) + 3](x)  # pool

        x = self.layer[3 * self.level + 1](x)
        x = self.relu(x)
        x = self.layer[3 * self.level + 2](x)
        x = self.relu(x)
        x = self.layer[3 * self.level + 3](x)  # dropout

        """Shallowing to reconstruct wavefields"""
        st_lvl = 3 * self.level + 4  # start from 3level+4
        for i in range(self.level):
            x = self.layer[st_lvl + i * 4 + 0](x)  # upsample
            x = self.layer[st_lvl + i * 4 + 1](x)  # conv1
            x = self.relu(x)

            if self.use_att:
                cat = self.attgates[-1 * (i + 1)](cat_content[-1 * (i + 1)], x)
            else:
                cat = cat_content[-1 * (i + 1)]
            x = torch.cat((cat, x), dim=1)

            x = self.layer[st_lvl + i * 4 + 2](x)  # conv2
            x = self.relu(x)
            x = self.layer[st_lvl + i * 4 + 3](x)  # conv3
            x = self.relu(x)
            
        x = self.layer[7 * self.level + 4](x)  # (None, 1, Nx, Nt) conv4

        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class MaxBlurPool2d(nn.Module):
    def __init__(self, nch, kernel_size=(2, 2)):
        """ must specify:
            Max pool
        """
        super(MaxBlurPool2d, self).__init__()
        self.kernel_size = kernel_size
        a = self.gaussion_filter(self.kernel_size[0])
        b = self.gaussion_filter(self.kernel_size[1])
        f = torch.matmul(a[:, None], b[None, :])

        f = f / torch.sum(f)
        f = f[None, None, :, :]
        f = f.repeat(nch, nch, 1, 1)

        pad1 = (kernel_size[0] - 1) // 2
        pad2 = kernel_size[0] - 1 - pad1
        pad3 = (kernel_size[1] - 1) // 2
        pad4 = kernel_size[1] - 1 - pad3
        pads = np.array([pad3, pad4, pad1, pad2])
        pads = torch.from_numpy(pads)
        filter = f.to(dtype=torch.float32)

        self.register_buffer('pads', pads)
        self.register_buffer('filter', filter)

    def forward(self, x):
        x = nn.MaxPool2d(kernel_size=self.kernel_size)(x)
        x = F.pad(x, self.pads.tolist(), 'constant', 0)
        x = F.conv2d(x, self.filter, stride=(1, 1), padding='valid')
        return x

    def gaussion_filter(self, kernel_size):

        if kernel_size == 1:
            f = torch.tensor([1., ])
        elif kernel_size == 2:
            f = torch.tensor([1., 1.])
        elif kernel_size == 3:
            f = torch.tensor([1., 2., 1.])
        elif kernel_size == 4:
            f = torch.tensor([1., 3., 3., 1.])
        elif kernel_size == 5:
            f = torch.tensor([1., 4., 6., 4., 1.])
        elif kernel_size == 6:
            f = torch.tensor([1., 5., 10., 10., 5., 1.])
        elif kernel_size == 7:
            f = torch.tensor([1., 6., 15., 20., 15., 6., 1.])
        return f


class AttentionGate(nn.Module):
    def __init__(self, nch):
        super(AttentionGate, self).__init__()
        self.conv1 = nn.Conv2d(nch, nch, (1, 1), padding=0)
        self.conv2 = nn.Conv2d(nch, nch, (1, 1), padding=0)
        self.conv3 = nn.Conv2d(nch, nch, (1, 1), padding=0)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, enc, dec):
        x = self.conv1(enc)
        y = self.conv2(dec)
        z = self.relu(x+y)
        z = self.sigmoid(self.conv3(z))

        return enc * z


class dataflow(nn.Module):

    def __init__(self, X, Nx_sub=1500, stride=750, mask_ratio=0.1, n_masks=10):
        """ This code assumes input size to be Ni, Nx=1500*n, Nt=1500；
            extract 1500^2 square samples and do masking in a bootstrap manner"""

        self.X = X  # DAS matrix
        self.Ni = X.shape[0]
        self.Nx = X.shape[1]
        self.Nt = X.shape[2]
        self.Nx_sub = Nx_sub  # Number of channels per sample
        self.stride = stride
        self.n_masks = n_masks  # number of times repeating the mask
        self.mask_traces = int(mask_ratio * Nx_sub)  # the number traces to mask
        self.__data_generation()

    def __len__(self):
        """ Number of samples """
        return int(self.n_masks * self.Ni * ((self.Nx - self.Nx_sub) / self.stride + 1))

    def __getitem__(self, idx):
        return (self.samples[idx], self.masks[idx]), self.masked_samples[idx]

    def __data_generation(self):
        X = self.X
        Ni = self.Ni
        Nt = self.Nt
        Nx = self.Nx
        Nx_sub = self.Nx_sub
        stride = self.stride
        n_masks = self.n_masks
        mask_traces = self.mask_traces

        n_total = self.__len__()  # total number of samples
        samples = np.zeros((n_total, 1, Nx_sub, Nt), dtype=np.float32)
        masks = np.ones_like(samples, dtype=np.float32)

        # Loop over samples
        for k in range(n_masks):
            for i in range(Ni):
                for j, st_ch in enumerate(np.arange(0, Nx-Nx_sub+1, stride)):
                    # %% slice each big image along channels
                    s = (k * Ni + i) * int((Nx-Nx_sub)//stride+1) + j
                    samples[s, 0, :, :] = X[i, st_ch:st_ch+Nx_sub, :]

                    rng = np.random.default_rng(s + 11)
                    trace_masked = rng.choice(Nx_sub, size=mask_traces, replace=False)
                    masks[s, 0, trace_masked, :] = masks[s, 0, trace_masked, :] * 0

        self.samples = samples
        self.masks = masks
        self.masked_samples = samples * (1 - masks)
        pass


class dataflow_nomask(nn.Module):

    def __init__(self, X, Nx_sub=1500, stride=750):
        """ This code assumes input size to be Ni, Nx=1500*n, Nt=1500；
            extract 1500^2 square samples"""

        self.X = X  # DAS matrix
        self.Ni = X.shape[0]
        self.Nx = X.shape[1]
        self.Nt = X.shape[2]
        self.Nx_sub = Nx_sub  # Number of channels per sample
        self.stride = stride
        self.__data_generation()

    def __len__(self):
        """ Number of samples """
        return int(self.Ni * ((self.Nx - self.Nx_sub) / self.stride + 1))

    def __getitem__(self, idx):
        return self.samples[idx]

    def __data_generation(self):
        X = self.X
        Ni = self.Ni
        Nt = self.Nt
        Nx = self.Nx
        Nx_sub = self.Nx_sub
        stride = self.stride

        n_total = self.__len__()  # total number of samples
        samples = np.zeros((n_total, 1, Nx_sub, Nt), dtype=np.float32)

        # Loop over samples
        for i in range(Ni):
            for j, st_ch in enumerate(np.arange(0, Nx-Nx_sub+1, stride)):
                # %% slice each big image along channels
                s = i * int((Nx-Nx_sub)//stride+1) + j
                samples[s, 0, :, :] = X[i, st_ch:st_ch+Nx_sub, :]
        self.samples = samples

        pass

class datalabel(nn.Module):

    def __init__(self, X, Y, Nx_sub=1500, stride=750, mask_ratio=0.1, n_masks=10):
        """ This code assumes input size to be Ni, Nx=1500*n, Nt=1500；
            extract 1500^2 square samples and do masking in a bootstrap manner"""

        self.X = X  # DAS matrix
        self.Y = Y  # DAS matrix
        self.Ni = X.shape[0]
        self.Nx = X.shape[1]
        self.Nt = X.shape[2]
        self.Nx_sub = Nx_sub  # Number of channels per sample
        self.stride = stride
        self.n_masks = n_masks  # number of times repeating the mask
        self.mask_traces = int(mask_ratio * Nx_sub)  # the number traces to mask
        self.__data_generation()

    def __len__(self):
        """ Number of samples """
        return int(self.n_masks * self.Ni * ((self.Nx - self.Nx_sub) / self.stride + 1))

    def __getitem__(self, idx):
        return (self.samples[idx], self.masks[idx]), self.masked_labels[idx]

    def __data_generation(self):
        X = self.X
        Y = self.Y
        Ni = self.Ni
        Nt = self.Nt
        Nx = self.Nx
        Nx_sub = self.Nx_sub
        stride = self.stride
        n_masks = self.n_masks
        mask_traces = self.mask_traces

        n_total = self.__len__()  # total number of samples
        samples = np.zeros((n_total, 1, Nx_sub, Nt), dtype=np.float32)
        labels = np.zeros((n_total, 1, Nx_sub, Nt), dtype=np.float32)
        masks = np.ones_like(samples, dtype=np.float32)

        # Loop over samples
        for k in range(n_masks):
            for i in range(Ni):
                for j, st_ch in enumerate(np.arange(0, Nx-Nx_sub+1, stride)):
                    # %% slice each big image along channels
                    s = (k * Ni + i) * int((Nx-Nx_sub)//stride+1) + j
                    samples[s, 0, :, :] = X[i, st_ch:st_ch + Nx_sub, :]
                    labels[s, 0, :, :] = Y[i, st_ch:st_ch + Nx_sub, :]

                    rng = np.random.default_rng(s + 11)
                    trace_masked = rng.choice(Nx_sub, size=mask_traces, replace=False)
                    masks[s, 0, trace_masked, :] = masks[s, 0, trace_masked, :] * 0

        self.samples = samples
        self.masks = masks
        self.masked_labels = labels * (1 - masks)
        pass
