#!/usr/bin/env python
#-*- coding: utf-8 -*-
import time
import random
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np

class HouseholderFlow(nn.Module):
    """
    Householder flow for q(z|x) modeling
    """
    def __init__(self, num_flow, hidden_size, code_size):
        super().__init__()
        self.num_flow = num_flow
        self.fc_hhv = nn.ModuleList()
        self.fc_hhv.append(nn.Linear(hidden_size, code_size))
        for t in range(1, self.num_flow):
            self.fc_hhv.append(nn.Linear(code_size, code_size))

    def hh_transform(self, v_nxt, zs):
        """
        # Householder transform.
        # Vector-vector product of v. Conceptually, v should be a column vector
        # (ignoring the batch dimension) so the result of v.v_transformed is a
        # matrix (i.e. not a dot product).
        #  - v_nxt size = [batch_size, h_len]
        #  - v_nxt.unsqueeze(2) size = [batch_size, h_len, 1]
        #  - v_nxt.unsqueeze(1) size = [batch_size, 1, h_len]
        #  - v_mult size = [batch_size, h_len, h_len]
        """
        v_mult = torch.matmul(v_nxt.unsqueeze(2), v_nxt.unsqueeze(1))
        # L2 norm squared of v, size = [batch_size, 1]
        v_l2_sq = torch.sum(v_nxt * v_nxt, 1).unsqueeze(1)
        z_nxt = zs - 2 * (torch.matmul(v_mult, zs.unsqueeze(2)).squeeze(2)) / v_l2_sq
        return z_nxt

    def forward(self, hidden, zs, v2=False):
        if v2:
            # hidden size: (batch_size x hidden_size)
            h = hidden
        else:
            # hidden size: tuple of (1 x batch_size x hidden_size)
            h = torch.cat(hidden, dim=2)[0].squeeze(0)
        if (self.num_flow > 0):
            v = [torch.zeros_like(h, requires_grad=True)] * self.num_flow
            z = [torch.zeros_like(zs, requires_grad=True)] * self.num_flow
            v[0] = h
            z[0] = zs
            for t in range(1, self.num_flow):
                v[t] = self.fc_hhv[t - 1](v[t - 1])
                z[t] = self.hh_transform(v[t], z[t - 1])
            return z[-1]
        else:
            return zs

# TODO: fix realnvp function ----> NOT correct dimension yet
class RealNVP(nn.Module):
    """
        nmt flow for q(z|x) modeling
    """
    def __init__(self, dim_in, dim_out):
        super(RealNVP, self).__init__()

        n_hid = 256
        n_mask = 6

        nets = lambda: nn.Sequential(
            nn.Linear(dim_in, n_hid),
            nn.LeakyReLU(),
            nn.BatchNorm1d(n_hid),
            nn.Linear(n_hid, n_hid),
            nn.LeakyReLU(),
            nn.BatchNorm1d(n_hid),
            nn.Linear(n_hid, dim_in),
            nn.Tanh())

        nett = lambda: nn.Sequential(
            nn.Linear(dim_in, n_hid),
            nn.LeakyReLU(),
            nn.BatchNorm1d(n_hid),
            nn.Linear(n_hid, n_hid),
            nn.LeakyReLU(),
            nn.BatchNorm1d(n_hid),
            nn.Linear(n_hid, dim_in))

        if dim_in > 2:
            masks = torch.from_numpy(np.random.randint(0, 2, (n_mask, dim_in)).astype(np.float32))
        else:
            masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))

        # prior = torch.distributions.MultivariateNormal(torch.zeros(dim_in).to(device), torch.eye(dim_in).to(device))

        self.mask = nn.Parameter(masks, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])
        self.linear = nn.Linear(dim_in, dim_out)

    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def forward(self, x):
        log_det_J, z = x.new_zeros(x.size()), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=0)
        return self.linear(z), log_det_J

    def log_prob(self, x, prior):
        z, logp = self.forward(x)
        return prior.log_prob(z) + logp

    def sample(self, bs):
        z = self.prior.sample((bs, 1))
        # logp = prior.log_prob(z)
        x = self.g(z)
        return x