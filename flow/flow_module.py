#!/usr/bin/env python
#-*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


'''
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from flow.nnet.weightnorm import Conv1dWeightNorm, LinearWeightNorm
from flow.nnet.attention import GlobalAttention, MultiHeadAttention
from flow.nnet.positional_encoding import PositionalEncoding
from flow.nnet.transformer import TransformerDecoderLayer
from utils import *

class Flow(nn.Module):
    """
    Normalizing Flow base class
    """
    _registry = dict()

    def __init__(self, inverse):
        super(Flow, self).__init__()
        self.inverse = inverse

    def forward(self, *inputs, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            *input: input [batch, *input_size]

        Returns: out: Tensor [batch, *input_size], logdet: Tensor [batch]
            out, the output of the flow
            logdet, the log determinant of :math:`\partial output / \partial input`
        """
        raise NotImplementedError

    def backward(self, *inputs, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            *input: input [batch, *input_size]

        Returns: out: Tensor [batch, *input_size], logdet: Tensor [batch]
            out, the output of the flow
            logdet, the log determinant of :math:`\partial output / \partial input`
        """
        raise NotImplementedError

    def init(self, *input, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def fwdpass(self, x: torch.Tensor, *h, init=False, init_scale=1.0, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x: Tensor
                The random variable before flow
            h: list of object
                other conditional inputs
            init: bool
                perform initialization or not (default: False)
            init_scale: float
                initial scale (default: 1.0)

        Returns: y: Tensor, logdet: Tensor
            y, the random variable after flow
            logdet, the log determinant of :math:`\partial y / \partial x`
            Then the density :math:`\log(p(y)) = \log(p(x)) - logdet`

        """
        if self.inverse:
            if init:
                raise RuntimeError('inverse flow shold be initialized with backward pass')
            else:
                return self.backward(x, *h, **kwargs)
        else:
            if init:
                return self.init(x, *h, init_scale=init_scale, **kwargs)
            else:
                return self.forward(x, *h, **kwargs)

    def bwdpass(self, y: torch.Tensor, *h, init=False, init_scale=1.0, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            y: Tensor
                The random variable after flow
            h: list of object
                other conditional inputs
            init: bool
                perform initialization or not (default: False)
            init_scale: float
                initial scale (default: 1.0)

        Returns: x: Tensor, logdet: Tensor
            x, the random variable before flow
            logdet, the log determinant of :math:`\partial x / \partial y`
            Then the density :math:`\log(p(y)) = \log(p(x)) + logdet`

        """
        if self.inverse:
            if init:
                return self.init(y, *h, init_scale=init_scale, **kwargs)
            else:
                return self.forward(y, *h, **kwargs)
        else:
            if init:
                raise RuntimeError('forward flow should be initialzed with forward pass')
            else:
                return self.backward(y, *h, **kwargs)

    @classmethod
    def register(cls, name: str):
        Flow._registry[name] = cls

    @classmethod
    def by_name(cls, name: str):
        return Flow._registry[name]

    @classmethod
    def from_params(cls, params: Dict):
        raise NotImplementedError

class ActNormFlow(Flow):
    def __init__(self, in_features, inverse=False):
        super(ActNormFlow, self).__init__(inverse)
        self.in_features = in_features
        self.log_scale = Parameter(torch.Tensor(in_features))
        self.bias = Parameter(torch.Tensor(in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.log_scale, mean=0, std=0.05)
        nn.init.constant_(self.bias, 0.)

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, N1, N2, ..., Nl, in_features]
            mask: Tensor
                mask tensor [batch, N1, N2, ...,Nl]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, N1, N2, ..., in_features], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        dim = input.dim()
        out = input * self.log_scale.exp() + self.bias
        out = out * mask.unsqueeze(dim - 1)
        logdet = self.log_scale.sum(dim=0, keepdim=True)
        if dim > 2:
            num = mask.view(out.size(0), -1).sum(dim=1)
            logdet = logdet * num
        return out, logdet

    def backward(self, input: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, N1, N2, ..., Nl, in_features]
            mask: Tensor
                mask tensor [batch, N1, N2, ...,Nl]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, N1, N2, ..., in_features], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        dim = input.dim()
        out = (input - self.bias) * mask.unsqueeze(dim - 1)
        out = out.div(self.log_scale.exp() + 1e-8)
        logdet = self.log_scale.sum(dim=0, keepdim=True) * -1.0
        if dim > 2:
            num = mask.view(out.size(0), -1).sum(dim=1)
            logdet = logdet * num
        return out, logdet

    def init(self, data: torch.Tensor, mask: torch.Tensor, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            data: input: Tensor
                input tensor [batch, N1, N2, ..., in_features]
            mask: Tensor
                mask tensor [batch, N1, N2, ...,Nl]
            init_scale: float
                initial scale

        Returns: out: Tensor , logdet: Tensor
            out: [batch, N1, N2, ..., in_features], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        with torch.no_grad():
            out, _ = self.forward(data, mask)
            mean = out.view(-1, self.in_features).mean(dim=0)
            std = out.view(-1, self.in_features).std(dim=0)
            inv_stdv = init_scale / (std + 1e-6)

            self.log_scale.add_(inv_stdv.log())
            self.bias.add_(-mean).mul_(inv_stdv)
            return self.forward(data, mask)

    def extra_repr(self):
        return 'inverse={}, in_features={}'.format(self.inverse, self.in_features)

    def from_params(cls, params: Dict) -> "ActNormFlow":
        return ActNormFlow(**params)

###############################################################################################
####################################### Coupling Blocks #######################################
###############################################################################################
class NICEConvBlock(nn.Module):
    def __init__(self, src_features, in_features, out_features, hidden_features, kernel_size, dropout=0.0):
        super(NICEConvBlock, self).__init__()
        self.conv1 = Conv1dWeightNorm(in_features, hidden_features, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv2 = Conv1dWeightNorm(hidden_features, hidden_features, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.activation = nn.ELU(inplace=True)
        self.attn = GlobalAttention(src_features, hidden_features, hidden_features, dropout=dropout)
        self.linear = LinearWeightNorm(hidden_features * 2, out_features, bias=True)

    def forward(self, x, mask, src, src_mask):
        """

        Args:
            x: Tensor
                input tensor [batch, length, in_features]
            mask: Tensor
                x mask tensor [batch, length]
            src: Tensor
                source input tensor [batch, src_length, src_features]
            src_mask: Tensor
                source mask tensor [batch, src_length]

        Returns: Tensor
            out tensor [batch, length, out_features]

        """
        out = self.activation(self.conv1(x.transpose(1, 2)))
        out = self.activation(self.conv2(out)).transpose(1, 2) * mask.unsqueeze(2)
        out = self.attn(out, src, key_mask=src_mask.eq(0))
        out = self.linear(torch.cat([x, out], dim=2))
        return out

    def init(self, x, mask, src, src_mask, init_scale=1.0):
        out = self.activation(self.conv1.init(x.transpose(1, 2), init_scale=init_scale))
        out = self.activation(self.conv2.init(out, init_scale=init_scale)).transpose(1, 2) * mask.unsqueeze(2)
        out = self.attn.init(out, src, key_mask=src_mask.eq(0), init_scale=init_scale)
        out = self.linear.init(torch.cat([x, out], dim=2), init_scale=0.0)
        return out


class NICERecurrentBlock(nn.Module):
    def __init__(self, rnn_mode, src_features, in_features, out_features, hidden_features, dropout=0.0):
        super(NICERecurrentBlock, self).__init__()
        if rnn_mode == 'RNN':
            RNN = nn.RNN
        elif rnn_mode == 'LSTM':
            RNN = nn.LSTM
        elif rnn_mode == 'GRU':
            RNN = nn.GRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)

        self.rnn = RNN(in_features, hidden_features // 2, batch_first=True, bidirectional=True)
        self.attn = GlobalAttention(src_features, hidden_features, hidden_features, dropout=dropout)
        self.linear = LinearWeightNorm(in_features + hidden_features, out_features, bias=True)

    def forward(self, x, mask, src, src_mask):
        lengths = mask.sum(dim=1).long()
        packed_out = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed_out)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=mask.size(1))
        # [batch, length, out_features]
        out = self.attn(out, src, key_mask=src_mask.eq(0))
        out = self.linear(torch.cat([x, out], dim=2))
        return out

    def init(self, x, mask, src, src_mask, init_scale=1.0):
        lengths = mask.sum(dim=1).long()
        packed_out = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed_out)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=mask.size(1))
        # [batch, length, out_features]
        out = self.attn.init(out, src, key_mask=src_mask.eq(0), init_scale=init_scale)
        out = self.linear.init(torch.cat([x, out], dim=2), init_scale=0.0)
        return out


class NICESelfAttnBlock(nn.Module):
    def __init__(self, src_features, in_features, out_features, hidden_features, heads, dropout=0.0,
                 pos_enc='add', max_length=100):
        super(NICESelfAttnBlock, self).__init__()
        assert pos_enc in ['add', 'attn']
        self.src_proj = nn.Linear(src_features, in_features, bias=False) if src_features != in_features else None
        self.pos_enc = PositionalEncoding(in_features, padding_idx=None, init_size=max_length + 1)
        self.pos_attn = MultiHeadAttention(in_features, heads, dropout=dropout) if pos_enc == 'attn' else None
        self.transformer = TransformerDecoderLayer(in_features, hidden_features, heads, dropout=dropout)
        self.linear = LinearWeightNorm(in_features, out_features, bias=True)

    def forward(self, x, mask, src, src_mask):
        if self.src_proj is not None:
            src = self.src_proj(src)

        key_mask = mask.eq(0)
        pos_enc = self.pos_enc(x) * mask.unsqueeze(2)
        if self.pos_attn is None:
            x = x + pos_enc
        else:
            x = self.pos_attn(pos_enc, x, x, key_mask)

        x = self.transformer(x, key_mask, src, src_mask.eq(0))
        return self.linear(x)

    def init(self, x, mask, src, src_mask, init_scale=1.0):
        if self.src_proj is not None:
            src = self.src_proj(src)

        key_mask = mask.eq(0)
        pos_enc = self.pos_enc(x) * mask.unsqueeze(2)
        if self.pos_attn is None:
            x = x + pos_enc
        else:
            x = self.pos_attn(pos_enc, x, x, key_mask)

        x = self.transformer.init(x, key_mask, src, src_mask.eq(0), init_scale=init_scale)
        x = x * mask.unsqueeze(2)
        return self.linear.init(x, init_scale=0.0)

###############################################################################################
####################################### Transform Blocks ######################################
###############################################################################################
class Transform():
    @staticmethod
    def fwd(z: torch.Tensor, mask: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    def bwd(z: torch.Tensor, mask: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class Additive(Transform):
    @staticmethod
    def fwd(z: torch.Tensor, mask: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = params
        z = (z + mu).mul(mask.unsqueeze(2))
        logdet = z.new_zeros(z.size(0))
        return z, logdet

    @staticmethod
    def bwd(z: torch.Tensor, mask: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = params
        z = (z - mu).mul(mask.unsqueeze(2))
        logdet = z.new_zeros(z.size(0))
        return z, logdet


class Affine(Transform):
    @staticmethod
    def fwd(z: torch.Tensor, mask: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_scale = params.chunk(2, dim=2)
        scale = log_scale.add_(2.0).sigmoid_()
        z = (scale * z + mu).mul(mask.unsqueeze(2))
        logdet = scale.log().mul(mask.unsqueeze(2)).view(z.size(0), -1).sum(dim=1)
        return z, logdet

    @staticmethod
    def bwd(z: torch.Tensor, mask: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_scale = params.chunk(2, dim=2)
        scale = log_scale.add_(2.0).sigmoid_()
        z = (z - mu).div(scale + 1e-12).mul(mask.unsqueeze(2))
        logdet = scale.log().mul(mask.unsqueeze(2)).view(z.size(0), -1).sum(dim=1) * -1.0
        return z, logdet


def arccosh(x):
    return torch.log(x + torch.sqrt(x.pow(2) - 1))


def arcsinh(x):
    return torch.log(x + torch.sqrt(x.pow(2) + 1))


class NLSQ(Transform):
    # A = 8 * math.sqrt(3) / 9 - 0.05  # 0.05 is a small number to prevent exactly 0 slope
    logA = np.log(8 * np.sqrt(3) / 9 - 0.05)  # 0.05 is a small number to prevent exactly 0 slope

    @staticmethod
    def get_pseudo_params(params):
        a, logb, cprime, logd, g = params.chunk(5, dim=2)

        # for stability
        logb = logb.mul_(0.4)
        cprime = cprime.mul_(0.3)
        logd = logd.mul_(0.4)

        # b = logb.add_(2.0).sigmoid_()
        # d = logd.add_(2.0).sigmoid_()
        # c = (NLSQ.A * b / d).mul(cprime.tanh_())

        c = (NLSQ.logA + logb - logd).exp_().mul(cprime.tanh_())
        b = logb.exp_()
        d = logd.exp_()
        return a, b, c, d, g

    @staticmethod
    def fwd(z: torch.Tensor, mask: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b, c, d, g = NLSQ.get_pseudo_params(params)

        arg = (d * z).add_(g)
        denom = arg.pow(2).add_(1)
        c = c / denom
        z = (b * z + a + c).mul(mask.unsqueeze(2))
        logdet = torch.log(b - 2 * c * d * arg / denom)
        logdet = logdet.mul(mask.unsqueeze(2)).view(z.size(0), -1).sum(dim=1)
        return z, logdet

    @staticmethod
    def bwd(z: torch.Tensor, mask: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b, c, d, g = NLSQ.get_pseudo_params(params)

        # double needed for stability. No effect on overall speed
        a = a.double()
        b = b.double()
        c = c.double()
        d = d.double()
        g = g.double()
        z = z.double()

        aa = -b * d.pow(2)
        bb = (z - a) * d.pow(2) - 2 * b * d * g
        cc = (z - a) * 2 * d * g - b * (1 + g.pow(2))
        dd = (z - a) * (1 + g.pow(2)) - c

        p = (3 * aa * cc - bb.pow(2)) / (3 * aa.pow(2))
        q = (2 * bb.pow(3) - 9 * aa * bb * cc + 27 * aa.pow(2) * dd) / (27 * aa.pow(3))

        t = -2 * torch.abs(q) / q * torch.sqrt(torch.abs(p) / 3)
        inter_term1 = -3 * torch.abs(q) / (2 * p) * torch.sqrt(3 / torch.abs(p))
        inter_term2 = 1 / 3 * arccosh(torch.abs(inter_term1 - 1) + 1)
        t = t * torch.cosh(inter_term2)

        tpos = -2 * torch.sqrt(torch.abs(p) / 3)
        inter_term1 = 3 * q / (2 * p) * torch.sqrt(3 / torch.abs(p))
        inter_term2 = 1 / 3 * arcsinh(inter_term1)
        tpos = tpos * torch.sinh(inter_term2)

        t[p > 0] = tpos[p > 0]
        z = t - bb / (3 * aa)
        arg = d * z + g
        denom = arg.pow(2) + 1
        logdet = torch.log(b - 2 * c * d * arg / denom.pow(2))

        z = z.float().mul(mask.unsqueeze(2))
        logdet = logdet.float().mul(mask.unsqueeze(2)).view(z.size(0), -1).sum(dim=1) * -1.0
        return z, logdet

###############################################################################################
########################### NICE with coupling & tranform blocks ##############################
###############################################################################################
class NICE(Flow):
    """
    NICE Flow
    """
    def __init__(self, src_features, features, hidden_features=None, inverse=False, split_dim=2, split_type='continuous', order='up', factor=2,
                 transform='affine', type='conv', kernel=3, rnn_mode='LSTM', heads=1, dropout=0.0, pos_enc='add', max_length=100):
        super(NICE, self).__init__(inverse)
        self.features = features
        assert split_dim in [1, 2]
        assert split_type in ['continuous', 'skip']
        if split_dim == 1:
            assert split_type == 'skip'
        if factor != 2:
            assert split_type == 'continuous'
        assert order in ['up', 'down']
        self.split_dim = split_dim
        self.split_type = split_type
        self.up = order == 'up'
        if split_dim == 2:
            out_features = features // factor
            in_features = features - out_features
            self.z1_channels = in_features if self.up else out_features
        else:
            in_features = features
            out_features = features
            self.z1_channels = None
        assert transform in ['additive', 'affine', 'nlsq']
        if transform == 'additive':
            self.transform = Additive
        elif transform == 'affine':
            self.transform = Affine
            out_features = out_features * 2
        elif transform == 'nlsq':
            self.transform = NLSQ
            out_features = out_features * 5
        else:
            raise ValueError('unknown transform: {}'.format(transform))

        if hidden_features is None:
            hidden_features = min(2 * in_features, 1024)
        assert type in ['conv', 'self_attn', 'rnn']
        if type == 'conv':
            self.net = NICEConvBlock(src_features, in_features, out_features, hidden_features, kernel_size=kernel, dropout=dropout)
        elif type == 'rnn':
            self.net = NICERecurrentBlock(rnn_mode, src_features, in_features, out_features, hidden_features, dropout=dropout)
        else:
            self.net = NICESelfAttnBlock(src_features, in_features, out_features, hidden_features,
                                         heads=heads, dropout=dropout, pos_enc=pos_enc, max_length=max_length)

    def split(self, z, mask):
        split_dim = self.split_dim
        split_type = self.split_type
        dim = z.size(split_dim)
        if split_type == 'continuous':
            return z.split([self.z1_channels, dim - self.z1_channels], dim=split_dim), mask
        elif split_type == 'skip':
            idx1 = torch.tensor(list(range(0, dim, 2))).to(z.device)
            idx2 = torch.tensor(list(range(1, dim, 2))).to(z.device)
            z1 = z.index_select(split_dim, idx1)
            z2 = z.index_select(split_dim, idx2)
            if split_dim == 1:
                mask = mask.index_select(split_dim, idx1)
            return (z1, z2), mask
        else:
            raise ValueError('unknown split type: {}'.format(split_type))

    def unsplit(self, z1, z2):
        split_dim = self.split_dim
        split_type = self.split_type
        if split_type == 'continuous':
            return torch.cat([z1, z2], dim=split_dim)
        elif split_type == 'skip':
            z = torch.cat([z1, z2], dim=split_dim)
            dim = z1.size(split_dim)
            idx = torch.tensor([i // 2 if i % 2 == 0 else i // 2 + dim for i in range(dim * 2)]).to(z.device)
            return z.index_select(split_dim, idx)
        else:
            raise ValueError('unknown split type: {}'.format(split_type))

    def calc_params(self, z: torch.Tensor, mask: torch.Tensor, src: torch.Tensor, src_mask: torch.Tensor):
        params = self.net(z, mask, src, src_mask)
        return params

    def init_net(self, z: torch.Tensor, mask: torch.Tensor, src: torch.Tensor, src_mask: torch.Tensor, init_scale=1.0):
        params = self.net.init(z, mask, src, src_mask, init_scale=init_scale)
        return params

    def forward(self, input: torch.Tensor, mask: torch.Tensor, src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input: Tensor
                input tensor [batch, length, in_features]
            mask: Tensor
                mask tensor [batch, length]
            src: Tensor
                source input tensor [batch, src_length, src_features]
            src_mask: Tensor
                source mask tensor [batch, src_length]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, length, in_features], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`
        """
        # [batch, length, in_channels]
        (z1, z2), mask = self.split(input, mask)
        # [batch, length, features]
        z, zp = (z1, z2) if self.up else (z2, z1)

        params = self.calc_params(z, mask, src, src_mask)
        zp, logdet = self.transform.fwd(zp, mask, params)

        z1, z2 = (z, zp) if self.up else (zp, z)
        return self.unsplit(z1, z2), logdet

    def backward(self, input: torch.Tensor, mask: torch.Tensor, src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input: Tensor
                input tensor [batch, length, in_features]
            mask: Tensor
                mask tensor [batch, length]
            src: Tensor
                source input tensor [batch, src_length, src_features]
            src_mask: Tensor
                source mask tensor [batch, src_length]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, length, in_features], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`
        """
        # [batch, length, in_channels]
        (z1, z2), mask = self.split(input, mask)
        # [batch, length, features]
        z, zp = (z1, z2) if self.up else (z2, z1)

        params = self.calc_params(z, mask, src, src_mask)
        zp, logdet = self.transform.bwd(zp, mask, params)

        z1, z2 = (z, zp) if self.up else (zp, z)
        return self.unsplit(z1, z2), logdet

    def init(self, data: torch.Tensor, mask: torch.Tensor, src: torch.Tensor, src_mask: torch.Tensor, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch, length, in_channels]
        (z1, z2), mask = self.split(data, mask)
        # [batch, length, features]
        z, zp = (z1, z2) if self.up else (z2, z1)

        params = self.init_net(z, mask, src, src_mask, init_scale=init_scale)
        zp, logdet = self.transform.fwd(zp, mask, params)

        z1, z2 = (z, zp) if self.up else (zp, z)
        return self.unsplit(z1, z2), logdet

    def extra_repr(self):
        return 'inverse={}, in_channels={}, scale={}'.format(self.inverse, self.in_channels, self.scale)

    @classmethod
    def from_params(cls, params: Dict) -> "NICE":
        return NICE(**params)

###############################################################################################
############################### Multihead Linear Flow block ###################################
###############################################################################################
class InvertibleLinearFlow(Flow):
    def __init__(self, in_features, inverse=False):
        super(InvertibleLinearFlow, self).__init__(inverse)
        self.in_features = in_features
        self.weight = Parameter(torch.Tensor(in_features, in_features))
        self.register_buffer('weight_inv', self.weight.data.clone())
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        self.sync()

    def sync(self):
        self.weight_inv.copy_(self.weight.data.inverse())

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, N1, N2, ..., Nl, in_features]
            mask: Tensor
                mask tensor [batch, N1, N2, ...,Nl]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, N1, N2, ..., in_features], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        dim = input.dim()
        # [batch, N1, N2, ..., in_features]
        out = F.linear(input, self.weight)
        _, logdet = torch.slogdet(self.weight)
        if dim > 2:
            num = mask.view(out.size(0), -1).sum(dim=1)
            logdet = logdet * num
        return out, logdet

    def backward(self, input: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, N1, N2, ..., Nl, in_features]
            mask: Tensor
                mask tensor [batch, N1, N2, ...,Nl]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, N1, N2, ..., in_features], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        dim = input.dim()
        # [batch, N1, N2, ..., in_features]
        out = F.linear(input, self.weight_inv)
        _, logdet = torch.slogdet(self.weight_inv)
        if dim > 2:
            num = mask.view(out.size(0), -1).sum(dim=1)
            logdet = logdet * num
        return out, logdet

    def init(self, data: torch.Tensor, mask: torch.Tensor, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.forward(data)

    def extra_repr(self):
        return 'inverse={}, in_features={}'.format(self.inverse, self.in_features)

    @classmethod
    def from_params(cls, params: Dict) -> "InvertibleLinearFlow":
        return InvertibleLinearFlow(**params)


class InvertibleMultiHeadFlow(Flow):
    @staticmethod
    def _get_heads(in_features):
        units = [32, 16, 8]
        for unit in units:
            if in_features % unit == 0:
                return in_features // unit
        assert in_features < 8, 'features={}'.format(in_features)
        return 1

    def __init__(self, in_features, heads=None, type='A', inverse=False):
        super(InvertibleMultiHeadFlow, self).__init__(inverse)
        self.in_features = in_features
        if heads is None:
            heads = InvertibleMultiHeadFlow._get_heads(in_features)
        self.heads = heads
        self.type = type
        assert in_features % heads == 0, 'features ({}) should be divided by heads ({})'.format(in_features, heads)
        assert type in ['A', 'B'], 'type should belong to [A, B]'
        self.weight = Parameter(torch.Tensor(in_features // heads, in_features // heads))
        self.register_buffer('weight_inv', self.weight.data.clone())
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        self.sync()

    def sync(self):
        self.weight_inv.copy_(self.weight.data.inverse())

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, N1, N2, ..., Nl, in_features]
            mask: Tensor
                mask tensor [batch, N1, N2, ...,Nl]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, N1, N2, ..., in_features], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        size = input.size()
        dim = input.dim()
        # [batch, N1, N2, ..., heads, in_features/ heads]
        if self.type == 'A':
            out = input.view(*size[:-1], self.heads, self.in_features // self.heads)
        else:
            out = input.view(*size[:-1], self.in_features // self.heads, self.heads).transpose(-2, -1)

        out = F.linear(out, self.weight)
        if self.type == 'B':
            out = out.transpose(-2, -1).contiguous()
        out = out.view(*size)

        _, logdet = torch.slogdet(self.weight)
        if dim > 2:
            num = mask.view(size[0], -1).sum(dim=1) * self.heads
            logdet = logdet * num
        return out, logdet

    def backward(self, input: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, N1, N2, ..., Nl, in_features]
            mask: Tensor
                mask tensor [batch, N1, N2, ...,Nl]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, N1, N2, ..., in_features], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        size = input.size()
        dim = input.dim()
        # [batch, N1, N2, ..., heads, in_features/ heads]
        if self.type == 'A':
            out = input.view(*size[:-1], self.heads, self.in_features // self.heads)
        else:
            out = input.view(*size[:-1], self.in_features // self.heads, self.heads).transpose(-2, -1)

        out = F.linear(out, self.weight_inv)
        if self.type == 'B':
            out = out.transpose(-2, -1).contiguous()
        out = out.view(*size)

        _, logdet = torch.slogdet(self.weight_inv)
        if dim > 2:
            num = mask.view(size[0], -1).sum(dim=1) * self.heads
            logdet = logdet * num
        return out, logdet

    def init(self, data: torch.Tensor, mask: torch.Tensor, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.forward(data, mask)

    def extra_repr(self):
        return 'inverse={}, in_features={}, heads={}, type={}'.format(self.inverse, self.in_features, self.heads, self.type)

    @classmethod
    def from_params(cls, params: Dict) -> "InvertibleMultiHeadFlow":
        return InvertibleMultiHeadFlow(**params)

###############################################################################################
####################################### NMT Flow ##############################################
###############################################################################################
class NMTFlowPOSAttnUnit(Flow):
    """
    Unit for POS Attention
    """
    def __init__(self, features, src_features, hidden_features=None, inverse=False,
                 transform='affine', heads=1, max_length=100, dropout=0.0):
        super(NMTFlowPOSAttnUnit, self).__init__(inverse)
        self.actnorm = ActNormFlow(features, inverse=inverse)
        self.coupling_up = NICE(src_features, features, hidden_features=hidden_features, inverse=inverse,
                                split_dim=2, split_type='continuous', order='up',
                                transform=transform, type='self_attn', heads=heads,
                                dropout=dropout, pos_enc='attn', max_length=max_length)

        self.coupling_down = NICE(src_features, features, hidden_features=hidden_features, inverse=inverse,
                                  split_dim=2, split_type='continuous', order='down',
                                  transform=transform, type='self_attn', heads=heads,
                                  dropout=dropout, pos_enc='add', max_length=max_length)

    def forward(self, input: torch.Tensor, tgt_mask: torch.Tensor,
                src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.actnorm.forward(input, tgt_mask)

        out, logdet = self.coupling_up.forward(out, tgt_mask, src, src_mask)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.coupling_down.forward(out, tgt_mask, src, src_mask)
        logdet_accum = logdet_accum + logdet

        return out, logdet_accum

    def backward(self, input: torch.Tensor, tgt_mask: torch.Tensor,
                 src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # block1 dim=2, type=continuous
        out, logdet_accum = self.coupling_down.backward(input, tgt_mask, src, src_mask)

        out, logdet = self.coupling_up.backward(out, tgt_mask, src, src_mask)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.actnorm.backward(out, tgt_mask)
        logdet_accum = logdet_accum + logdet

        return out, logdet_accum

    def init(self, data: torch.Tensor, tgt_mask: torch.Tensor,
             src: torch.Tensor, src_mask: torch.Tensor, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.actnorm.init(data, tgt_mask, init_scale=init_scale)

        out, logdet = self.coupling_up.init(out, tgt_mask, src, src_mask, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.coupling_down.init(out, tgt_mask, src, src_mask, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        return out, logdet_accum


class NMTFlowUnit(Flow):
    """
    One Unit of NMTFlowStep
    """

    def __init__(self, features, src_features, hidden_features=None, inverse=False, transform='affine',
                 coupling_type='conv', kernel_size=3, rnn_mode='LSTM', heads=1, max_length=100,
                 dropout=0.0, split_timestep=True):
        super(NMTFlowUnit, self).__init__(inverse)
        # dim=2, type=continuous
        self.coupling1_up = NICE(src_features, features, hidden_features=hidden_features, inverse=inverse,
                                 split_dim=2, split_type='continuous', order='up',
                                 transform=transform, type=coupling_type, kernel=kernel_size, rnn_mode=rnn_mode,
                                 heads=heads, dropout=dropout, pos_enc='add', max_length=max_length)

        self.coupling1_down = NICE(src_features, features, hidden_features=hidden_features, inverse=inverse,
                                   split_dim=2, split_type='continuous', order='down',
                                   transform=transform, type=coupling_type, kernel=kernel_size, rnn_mode=rnn_mode,
                                   heads=heads, dropout=dropout, pos_enc='add', max_length=max_length)
        self.actnorm1 = ActNormFlow(features, inverse=inverse)

        # dim=2, type=skip
        self.coupling2_up = NICE(src_features, features, hidden_features=hidden_features, inverse=inverse,
                                 split_dim=2, split_type='skip', order='up',
                                 transform=transform, type=coupling_type, kernel=kernel_size, rnn_mode=rnn_mode,
                                 heads=heads, dropout=dropout, pos_enc='add', max_length=max_length)

        self.coupling2_down = NICE(src_features, features, hidden_features=hidden_features, inverse=inverse,
                                   split_dim=2, split_type='skip', order='down',
                                   transform=transform, type=coupling_type, kernel=kernel_size, rnn_mode=rnn_mode,
                                   heads=heads, dropout=dropout, pos_enc='add', max_length=max_length)

        self.split_timestep = split_timestep
        if split_timestep:
            self.actnorm2 = ActNormFlow(features, inverse=inverse)
            # dim=1, type=skip
            self.coupling3_up = NICE(src_features, features, hidden_features=hidden_features, inverse=inverse,
                                     split_dim=1, split_type='skip', order='up',
                                     transform=transform, type=coupling_type, kernel=kernel_size, rnn_mode=rnn_mode,
                                     heads=heads, dropout=dropout, pos_enc='add', max_length=max_length)

            self.coupling3_down = NICE(src_features, features, hidden_features=hidden_features, inverse=inverse,
                                       split_dim=1, split_type='skip', order='down',
                                       transform=transform, type=coupling_type, kernel=kernel_size, rnn_mode=rnn_mode,
                                       heads=heads, dropout=dropout, pos_enc='add', max_length=max_length)
        else:
            self.actnorm2 = None
            self.coupling3_up = None
            self.coupling3_down = None

    def forward(self, input: torch.Tensor, tgt_mask: torch.Tensor,
                src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # block1 dim=2, type=continuous
        out, logdet_accum = self.coupling1_up.forward(input, tgt_mask, src, src_mask)

        out, logdet = self.coupling1_down.forward(out, tgt_mask, src, src_mask)
        logdet_accum = logdet_accum + logdet

        # ================================================================================

        out, logdet = self.actnorm1.forward(out, tgt_mask)
        logdet_accum = logdet_accum + logdet

        # ================================================================================

        # block2 dim=2, type=skip
        out, logdet = self.coupling2_up.forward(out, tgt_mask, src, src_mask)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.coupling2_down.forward(out, tgt_mask, src, src_mask)
        logdet_accum = logdet_accum + logdet

        if self.split_timestep:
            # ================================================================================

            out, logdet = self.actnorm2.forward(out, tgt_mask)
            logdet_accum = logdet_accum + logdet

            # ================================================================================

            # block3 dim=1, type=skip
            out, logdet = self.coupling3_up.forward(out, tgt_mask, src, src_mask)
            logdet_accum = logdet_accum + logdet

            out, logdet = self.coupling3_down.forward(out, tgt_mask, src, src_mask)
            logdet_accum = logdet_accum + logdet

        return out, logdet_accum

    def backward(self, input: torch.Tensor, tgt_mask: torch.Tensor,
                 src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.split_timestep:
            # block3 dim=1, type=skip
            out, logdet_accum = self.coupling3_down.backward(input, tgt_mask, src, src_mask)

            out, logdet = self.coupling3_up.backward(out, tgt_mask, src, src_mask)
            logdet_accum = logdet_accum + logdet

            # ================================================================================

            out, logdet = self.actnorm2.backward(out, tgt_mask)
            logdet_accum = logdet_accum + logdet

            # ================================================================================
        else:
            out, logdet_accum = input, 0

        # block2 dim=2, type=skip
        out, logdet = self.coupling2_down.backward(out, tgt_mask, src, src_mask)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.coupling2_up.backward(out, tgt_mask, src, src_mask)
        logdet_accum = logdet_accum + logdet

        # ================================================================================

        out, logdet = self.actnorm1.backward(out, tgt_mask)
        logdet_accum = logdet_accum + logdet

        # ================================================================================

        # block1 dim=2, type=continuous
        out, logdet = self.coupling1_down.backward(out, tgt_mask, src, src_mask)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.coupling1_up.backward(out, tgt_mask, src, src_mask)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    def init(self, data: torch.Tensor, tgt_mask: torch.Tensor,
             src: torch.Tensor, src_mask: torch.Tensor, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # block1 dim=2, type=continuous
        out, logdet_accum = self.coupling1_up.init(data, tgt_mask, src, src_mask, init_scale=init_scale)

        out, logdet = self.coupling1_down.init(out, tgt_mask, src, src_mask, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        # ================================================================================

        out, logdet = self.actnorm1.init(out, tgt_mask, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        # ================================================================================

        # block2 dim=2, type=skip
        out, logdet = self.coupling2_up.init(out, tgt_mask, src, src_mask, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.coupling2_down.init(out, tgt_mask, src, src_mask, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        if self.split_timestep:
            # ================================================================================

            out, logdet = self.actnorm2.init(out, tgt_mask, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet

            # ================================================================================

            # block3 dim=1, type=skip
            out, logdet = self.coupling3_up.init(out, tgt_mask, src, src_mask, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet

            out, logdet = self.coupling3_down.init(out, tgt_mask, src, src_mask, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet

        return out, logdet_accum


class NMTFlowStep(Flow):
    """
    One step of NMTFlow
    """

    def __init__(self, features, src_features, hidden_features=None, inverse=False, transform='affine',
                 coupling_type='conv', kernel_size=3, rnn_mode='LSTM', heads=1, max_length=100,
                 dropout=0.0, split_timestep=True):
        super(NMTFlowStep, self).__init__(inverse)
        self.actnorm1 = ActNormFlow(features, inverse=inverse)
        self.linear1 = InvertibleMultiHeadFlow(features, type='A', inverse=inverse)
        self.unit1 = NMTFlowUnit(features, src_features, hidden_features=hidden_features, inverse=inverse,
                                 transform=transform, coupling_type=coupling_type, kernel_size=kernel_size, rnn_mode=rnn_mode,
                                 heads=heads, dropout=dropout, max_length=max_length, split_timestep=split_timestep)
        self.actnorm2 = ActNormFlow(features, inverse=inverse)
        self.linear2 = InvertibleMultiHeadFlow(features, type='B', inverse=inverse)
        self.unit2 = NMTFlowUnit(features, src_features, hidden_features=hidden_features, inverse=inverse,
                                 transform=transform, coupling_type=coupling_type, kernel_size=kernel_size, rnn_mode=rnn_mode,
                                 heads=heads, dropout=dropout, max_length=max_length, split_timestep=split_timestep)

    def sync(self):
        self.linear1.sync()
        self.linear2.sync()

    def forward(self, input: torch.Tensor, tgt_mask: torch.Tensor,
                src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.actnorm1.forward(input, tgt_mask)

        out, logdet = self.linear1.forward(out, tgt_mask)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.unit1.forward(out, tgt_mask, src, src_mask)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.actnorm2.forward(out, tgt_mask)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.linear2.forward(out, tgt_mask)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.unit2.forward(out, tgt_mask, src, src_mask)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    def backward(self, input: torch.Tensor, tgt_mask: torch.Tensor,
                 src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.unit2.backward(input, tgt_mask, src, src_mask)

        out, logdet = self.linear2.backward(out, tgt_mask)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.actnorm2.backward(out, tgt_mask)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.unit1.backward(out, tgt_mask, src, src_mask)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.linear1.backward(out, tgt_mask)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.actnorm1.backward(out, tgt_mask)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    def init(self, data: torch.Tensor, tgt_mask: torch.Tensor,
             src: torch.Tensor, src_mask: torch.Tensor, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.actnorm1.init(data, tgt_mask, init_scale=init_scale)

        out, logdet = self.linear1.init(out, tgt_mask, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.unit1.init(out, tgt_mask, src, src_mask, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.actnorm2.init(out, tgt_mask, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.linear2.init(out, tgt_mask, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.unit2.init(out, tgt_mask, src, src_mask, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum


class NMTFlowBlock(Flow):
    """
    NMT Flow Block
    """

    def __init__(self, num_steps, features, src_features, hidden_features=None, inverse=False, prior=False, factor=2,
                 transform='affine', coupling_type='conv', kernel_size=3, rnn_mode='LSTM', heads=1, max_length=100,
                 dropout=0.0, pos_attn=False):
        super(NMTFlowBlock, self).__init__(inverse)
        if pos_attn:
            self.pos_attn = NMTFlowPOSAttnUnit(features, src_features,hidden_features=hidden_features,
                                               inverse=inverse, transform=transform, heads=heads,
                                               max_length=max_length, dropout=dropout)
        else:
            self.pos_attn = None

        steps = [NMTFlowStep(features, src_features, hidden_features=hidden_features, inverse=inverse,
                             transform=transform, coupling_type=coupling_type, kernel_size=kernel_size,
                             rnn_mode=rnn_mode, heads=heads, max_length=max_length,
                             dropout=dropout, split_timestep=prior) for _ in range(num_steps)]
        self.steps = nn.ModuleList(steps)
        if prior:
            assert features % factor == 0, 'features {} should divide factor {}'.format(features, factor)
            self.prior = NICE(src_features, features, hidden_features=hidden_features, inverse=inverse,
                              split_dim=2, split_type='continuous', order='up', factor=factor,
                              transform=transform, type=coupling_type, kernel=kernel_size,
                              heads=heads, rnn_mode=rnn_mode, pos_enc='add', max_length=max_length, dropout=dropout)
            self.z_features = features - features // factor
            assert self.z_features == self.prior.z1_channels
        else:
            self.prior = None
            self.z_features = features

    def sync(self):
        for step in self.steps:
            step.sync()

    def forward(self, input: torch.Tensor, tgt_mask: torch.Tensor,
                src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch]
        if self.pos_attn is None:
            logdet_accum = input.new_zeros(input.size(0))
            out = input
        else:
            out, logdet_accum = self.pos_attn.forward(input, tgt_mask, src, src_mask)

        for step in self.steps:
            out, logdet = step.forward(out, tgt_mask, src, src_mask)
            logdet_accum = logdet_accum + logdet

        if self.prior is not None:
            out, logdet = self.prior.forward(out, tgt_mask, src, src_mask)
            logdet_accum = logdet_accum + logdet

        return out, logdet_accum

    def backward(self, input: torch.Tensor, tgt_mask: torch.Tensor,
                 src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.prior is None:
            logdet_accum = input.new_zeros(input.size(0))
            out = input
        else:
            out, logdet_accum = self.prior.backward(input, tgt_mask, src, src_mask)

        for step in reversed(self.steps):
            out, logdet = step.backward(out, tgt_mask, src, src_mask)
            logdet_accum = logdet_accum + logdet

        if self.pos_attn is not None:
            out, logdet = self.pos_attn.backward(out, tgt_mask, src, src_mask)
            logdet_accum = logdet_accum + logdet

        return out, logdet_accum

    @overrides
    def init(self, data: torch.Tensor, tgt_mask: torch.Tensor,
             src: torch.Tensor, src_mask: torch.Tensor, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch]
        if self.pos_attn is None:
            logdet_accum = data.new_zeros(data.size(0))
            out = data
        else:
            out, logdet_accum = self.pos_attn.init(data, tgt_mask, src, src_mask, init_scale=init_scale)

        for step in self.steps:
            out, logdet = step.init(out, tgt_mask, src, src_mask, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet

        if self.prior is not None:
            out, logdet = self.prior.init(out, tgt_mask, src, src_mask, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet

        return out, logdet_accum


class NMTFlow(Flow):
    """
    NMT Flow
    """

    def __init__(self, levels, num_steps, features, src_features, factors, hidden_features=None, inverse=False,
                 transform='affine', coupling_type='conv', kernel_size=3, rnn_mode='LSTM', heads=1, pos_enc='add', max_length=100, dropout=0.0):
        super(NMTFlow, self).__init__(inverse)
        assert levels == len(num_steps)
        assert levels == len(factors) + 1
        blocks = []
        self.levels = levels
        self.features = features
        pos_attn = coupling_type == 'self_attn' and pos_enc == 'attn'

        for level in range(levels):
            if level == levels - 1:
                block = NMTFlowBlock(num_steps[level], features, src_features, hidden_features=hidden_features,
                                     inverse=inverse, prior=False, coupling_type=coupling_type, transform=transform,
                                     kernel_size=kernel_size, rnn_mode=rnn_mode, heads=heads, max_length=max_length,
                                     dropout=dropout, pos_attn=pos_attn)
            else:
                factor = factors[level]
                block = NMTFlowBlock(num_steps[level], features, src_features, hidden_features=hidden_features,
                                     inverse=inverse, prior=True, factor=factor, coupling_type=coupling_type,
                                     transform=transform,kernel_size=kernel_size, rnn_mode=rnn_mode, heads=heads,
                                     max_length=max_length, dropout=dropout, pos_attn=pos_attn)
                features = block.z_features * 2
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

    def sync(self):
        for block in self.blocks:
            block.sync()

    def forward(self, input: torch.Tensor, tgt_mask: torch.Tensor,
                src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = input.new_zeros(input.size(0))
        out = input
        outputs = []
        for i, block in enumerate(self.blocks):
            out, logdet = block.forward(out, tgt_mask, src, src_mask)
            logdet_accum = logdet_accum + logdet
            if i < self.levels - 1:
                out1, out2 = split(out, block.z_features)
                outputs.append(out2)
                out, tgt_mask = squeeze(out1, tgt_mask)

        for _ in range(self.levels - 1):
            out2 = outputs.pop()
            out = unsqueeze(out)
            out = unsplit([out, out2])
        assert len(outputs) == 0
        return out, logdet_accum

    def backward(self, input: torch.Tensor, tgt_mask: torch.Tensor,
                 src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        masks = []
        out = input
        for i in range(self.levels - 1):
            out1, out2 = split(out, self.blocks[i].z_features)
            outputs.append(out2)
            masks.append(tgt_mask)
            out, tgt_mask = squeeze(out1, tgt_mask)

        logdet_accum = input.new_zeros(input.size(0))
        for i, block in enumerate(reversed(self.blocks)):
            if i > 0:
                out2 = outputs.pop()
                tgt_mask = masks.pop()
                out1 = unsqueeze(out)
                out = unsplit([out1, out2])
            out, logdet = block.backward(out, tgt_mask, src, src_mask)
            logdet_accum = logdet_accum + logdet
        assert len(outputs) == 0
        assert len(masks) == 0

        return out, logdet_accum

    def init(self, data: torch.Tensor, tgt_mask: torch.Tensor,
             src: torch.Tensor, src_mask: torch.Tensor, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = data.new_zeros(data.size(0))
        out = data
        outputs = []
        for i, block in enumerate(self.blocks):
            out, logdet = block.init(out, tgt_mask, src, src_mask, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet
            if i < self.levels - 1:
                out1, out2 = split(out, block.z_features)
                outputs.append(out2)
                out, tgt_mask = squeeze(out1, tgt_mask)

        for _ in range(self.levels - 1):
            out2 = outputs.pop()
            out = unsqueeze(out)
            out = unsplit([out, out2])
        assert len(outputs) == 0
        return out, logdet_accum

    @classmethod
    def from_params(cls, params: Dict) -> "NMTFlow":
        return NMTFlow(**params)
'''