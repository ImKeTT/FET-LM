#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: loss.py
@author: ImKe at 2021/3/21
@feature: #Enter features here
@scenario: #Enter scenarios here
"""
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import LogNormal
from torch.distributions import kl_divergence

import numpy as np


def seq_recon_loss(outputs, targets, pad_id):
    return F.cross_entropy(
        outputs.view(-1, outputs.size(2)),
        targets.view(-1),
        size_average=False, ignore_index=pad_id)


def bow_recon_loss(outputs, targets):
    """
    Note that outputs is the bag-of-words log likelihood predictions.
    targets is the target counts.
    """
    # outputs: [batch_size x vocab_size]
    return - torch.sum(targets * outputs)

def seq_recon_loss_from_zt(outputs, targets, zt=None):
    """
    :param outputs: BoW likelihood predictions(without log!) 
    :param targets: target counts
    :return: reconstruction loss
    """
    ########### Normalize everything ###########
    outputs /= torch.sum(outputs)
    targets /= torch.sum(targets)
    if zt is None:
        return torch.abs(torch.sum(outputs - targets))
    else:
        # todo: zt reconstruction implementation
        return 0

def discriminator_loss(discri_log, bow_target):
    """Calculate discriminator loss in BoW manner"""
    # batch_size = bow_target.size(0)
    # vocab_size = discri_log.size(-1)
    # discri_log = discri_log.view(batch_size, -1, vocab_size)
    # discri_log = torch.sum(discri_log, dim=1) / torch.full((batch_size, vocab_size), discri_log.size(1)).to(bow_target.device)
    return -torch.sum(bow_target * discri_log)

def standard_prior_like(posterior):
    loc = torch.zeros_like(posterior.loc)
    scale = torch.ones_like(posterior.scale)
    Dist = type(posterior)
    return Dist(loc, scale)


def normal_kld(posterior, prior=None):
    if prior is None:
        prior = standard_prior_like(posterior)
    return torch.sum(kl_divergence(posterior, prior))

def cal_tc(D_z, D_z_per):
    batch_size = D_z.size(0)
    device = D_z.device
    ones = torch.ones(batch_size, dtype=torch.long, device=device)
    zeros = torch.zeros(batch_size, dtype=torch.long, device=device)

    return 0.5*(F.cross_entropy(D_z, ones) + F.cross_entropy(D_z_per, zeros))

# todo: is hhf_kld the right KLD metric ?
def hhf_kld(z_start, z_end):
    """
    KL term if flow is used for modeling posterior q(z|x)
    """
    # Log of posterior distribution E_{q(z|x)}log[q(z|x)] at the end of flow (zero-mean Gaussian with
    # full covariance matrix). Constant log(2*pi) has been dropped since it'll
    # get cancelled out in KL divergence anyway.
    log_q = -0.5 * torch.sum(torch.pow(z_end, 2), 1)
    # Log of approx. prior. E_{q(z|x)}log[p(z)] at the start of flow (Gaussian with diagonal
    # covariance matrix). Constant log(2*pi) has been dropped, here we assume q(z|x) is normal
    # gaussian distribution for simplicity.
    mean = torch.zeros_like(z_start)
    var = torch.ones_like(z_start)
    # 1. assume q(z_s) ~ N(0, I)
    # 2. or assume p(z_s) with full covariance matrix
    log_p = -0.5 * torch.sum(torch.log(var) + torch.pow(z_start - mean, 2) / var, 1)
    # KL divergence
    # REVISIT: KL divergence should be the expectation of (log_q - log_p) w.r.t. q(z)
    # REVISIT: The official implementation seems to use Monte Carlo estimate of the
    # REVISIT: expectation with a sample size of 1. (This may sound very inaccurate
    # REVISIT: but the reconstruction error is also using Monte Carlo estimate of
    # REVISIT: expectation with just 1 sample)
    KL_loss = torch.abs(torch.sum((log_q - log_p)))
    # KL_loss = torch.sum((log_q - log_p))

    return KL_loss

def nmt_kld(logp, z_end):
    """
    KL term if flow is used for modeling posterior q(z|x)
    """
    device = z_end.device
    mean = torch.zeros(z_end.size(1)).to(device)
    var = torch.eye(z_end.size(1)).to(device)
    prior = MultivariateNormal(mean, var)
    return -prior.log_prob(z_end) - logp

def compute_kernel(z, z_sample):
    kernel_input = (z.unsqueeze(1) - z_sample.unsqueeze(0)).pow(2).mean(2) / z.shape[-1]
    return torch.exp(-kernel_input) # (x_size, y_size)
def mmd(z, z_sample):
    """
    distribution distance by MMD method
    :param z: posterior z
    :param z_sample: z sampled for normal distribution
    :return: distribution distance
    """
    zz = compute_kernel(z, z)
    ss = compute_kernel(z_sample, z_sample)
    zs = compute_kernel(z, z_sample)
    return zz.mean() + ss.mean() - 2 * zs.mean()

def compute_kernel_v2(x, y):
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]

    tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
    tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)
    return torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2) / dim * 1.0)
def mmd_v2(x, y):
    x_kernel = compute_kernel_v2(x, x)
    y_kernel = compute_kernel_v2(y, y)
    xy_kernel = compute_kernel_v2(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
