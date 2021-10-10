#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: topic_model.py
@author: ImKe at 2021/3/21
@feature: #Enter features here
@scenario: #Enter scenarios here
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.distributions import Normal, LogNormal, Dirichlet
from flow.flow_layer import HouseholderFlow, RealNVP
from utils import batch_beam_search_decoding, beam_decode


class Seq2Bow(nn.Module):
    """Converts sequences of tokens to bag of words representations. """
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, inputs, ignore_index):
        # inputs dim: batch_size x max_len
        bow = torch.zeros(
            (inputs.size(0), self.vocab_size),
            dtype=torch.float,
            device=inputs.device
        )
        ones = torch.ones_like(
            inputs, dtype=torch.float,
        )
        bow.scatter_add_(1, inputs, ones)
        bow[:, ignore_index] = 0
        # return bow_target: target Bag of Word representation
        return bow

# todo: 1. bidirectional seqencoder 2. apply GRU cell 3. dynamiclly fuse zs,zt
class SeqEncoder(nn.Module):
    """Sequence encoder. Used to calculate q(z|x). """

    def __init__(self, input_size, hidden_size, dropout, BR=True, cell='lstm'):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        if cell == 'lstm':
            self.rnn = nn.LSTM(
                input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=BR
            )
        elif cell=='gru':
            self.rnn = nn.GRU(
                input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=BR
            )
        else:
            raise Exception('no such rnn cell')

    def forward(self, inputs, lengths):
        inputs = pack(
            self.drop(inputs), lengths, batch_first=True
        )
        _, hn = self.rnn(inputs)
        return hn


class Seq2hidden_t(nn.Module):
    """
    Seqence encoder. Calculate q(z_t|x_seq)
    """
    def __init__(self, vocab_size, hidden_size, c_length, num_topic):
        super().__init__()
        # each dimension of z_t represents a latent topic
        self.fc1 = nn.Linear(vocab_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.fcmu = nn.Linear(c_length, num_topic)
        self.fclv = nn.Linear(c_length, num_topic)
        self.bnmu = nn.BatchNorm1d(num_topic)
        self.bnlv = nn.BatchNorm1d(num_topic)
        self.c_len = c_length
        self.num_topic = num_topic
        self.W_t_layer = nn.Linear(num_topic, num_topic, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.MaxPool1d(kernel_size=num_topic)

    def forward(self, inputs):
        # inputs is bow of size: (batch_size x vocab_size)
        h1 = F.softplus(self.fc1(inputs))
        h2 = F.softplus(self.fc2(h1))
        c = self.pool(h2.unsqueeze(0)).view(inputs.size(0), -1)
        # [batch_size x z_t size]
        mu = self.bnmu(self.fcmu(c))
        lv = self.bnlv(self.fclv(c))
        # noise = torch.randn([inputs.size(0), self.num_topic])
        # [batch_size x z_t size]
        # p = mu + noise * (0.5 * lv).exp()
        dist = LogNormal(mu, (0.5 * lv).exp())
        return dist

class Hidden_t2Seq(nn.Module):
    """
    Seqence decoder from topic latent. Calculate p(x_seq|z_t)
    """
    def __init__(self, num_topic, vocab_size, dropout):
        super().__init__()
        self.fc = nn.Linear(num_topic, vocab_size)
        self.bn = nn.BatchNorm1d(vocab_size)
        self.drop = nn.Dropout(dropout)
    def forward(self, zt):
        zt = self.drop(zt)
        return self.bn(self.fc(zt))


# > encoder in our topic modeling component act as the discriminator
class BowEncoder(nn.Module):
    """Bag of words encoder. Used to calculate q(t|x). """

    def __init__(self, vocab_size, hidden_size, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(vocab_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs):
        # inputs is bow of size: (batch_size x vocab_size)
        h1 = F.softplus(self.fc1(inputs))
        h2 = F.softplus(self.fc2(h1))
        h = self.drop(h2)
        return h


class Hidden2Normal(nn.Module):
    """
    Converts hidden state from the SeqEncoder to normal
    distribution. Calculates q(z|x).

    """

    def __init__(self, hidden_size, code_size):
        super().__init__()
        self.fcmu = nn.Linear(hidden_size * 2, code_size)
        self.fclv = nn.Linear(hidden_size * 2, code_size)
        self.bnmu = nn.BatchNorm1d(code_size)
        self.bnlv = nn.BatchNorm1d(code_size)

    def reparameterize(self, mean, logvar):
        sd = torch.exp(0.5 * logvar)  # Standard deviation
        # We'll assume the posterior is a multivariate Gaussian
        eps = torch.randn_like(sd)
        z = eps.mul(sd).add(mean)
        return z

    def forward(self, hidden):
        # hidden size: tuple of (1 x batch_size x hidden_size)
        h = torch.cat(hidden, dim=2)[0].squeeze(0)
        mu = self.bnmu(self.fcmu(h))
        lv = self.bnlv(self.fclv(h))
        dist = Normal(mu, (0.5 * lv).exp())
        # sample_z = self.reparameterize(mu, lv)
        sample_z = dist.rsample()
        # dist: for normal posterior modeling
        # z: for loss calculation (flow posterior modeling)
        # mu: for loss calculation (flow posterior modeling) [batch_size x code_size]
        # lv: for loss calculation (flow posterior modeling)
        return dist, sample_z


class topic_Hidden2Normal(nn.Module):
    """
    Converts hidden state from BowEncoder to log-normal
    distribution. Calculates q(t|x,z) = q(t|x)
    each topic = one Guassian distribution

    """

    def __init__(self, hidden_size, num_topics, device):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.fcmu = nn.Linear(hidden_size, num_topics)
        self.fclv = nn.Linear(hidden_size, num_topics)
        self.var_embedding = nn.Parameter(torch.zeros((hidden_size, num_topics)))
        self.var_linear = nn.Linear(num_topics, hidden_size)
        self.bnmu = nn.BatchNorm1d(num_topics)
        self.bnlv = nn.BatchNorm1d(num_topics)

    def encode_var(self, inputs, return_p=False):
        logits = self.var_linear(inputs)
        prob = F.softmax(logits, -1)
        if return_p:
            return torch.matmul(prob, self.var_embedding), prob
        return torch.matmul(prob, self.var_embedding)

    def orthogonal_regularizer(self, norm=100):
        tmp = torch.mm(self.var_embedding, self.var_embedding.permute(1, 0))
        return torch.norm(tmp - norm * torch.diag(torch.ones(self.hidden_size, device=self.device)), 2)

    def forward(self, hidden, return_origin=False):
        # hidden size: (batch_size x hidden_size)
        lv = self.bnlv(self.fclv(hidden))
        mu = self.bnmu(self.fcmu(hidden))
        if not return_origin:
            mu = self.encode_var(mu)
        print("finish t mu")
        return LogNormal(mu, (0.5 * lv).exp()), mu


class topic_Hidden2Dirichlet(nn.Module):
    """
    Converts hidden state from BowEncoder to dirichlet
    distribution. Calculates q(t|x,z) = q(t|x)

    """

    def __init__(self, hidden_size, num_topics):
        super().__init__()
        self.fc = nn.Linear(hidden_size, num_topics)
        self.bn = nn.BatchNorm1d(num_topics)

    def encode_var(self, inputs, return_p=False):
        pass

    def orthogonal_regularizer(self, norm=100):
        pass

    def forward(self, hidden, return_origin=True):
        # hidden size: (batch_size x hidden_size)
        alphas = self.bn(self.fc(hidden)).exp().cpu()
        # Dirichlet only supports cpu backprop for now
        dist = Dirichlet(alphas)
        return dist, alphas


class Code2LogNormal(nn.Module):
    """Calculates p(t|z). """

    def __init__(self, code_size, hidden_size, num_topics):
        super().__init__()
        self.fc1 = nn.Linear(code_size, hidden_size)
        self.fcmu = nn.Linear(hidden_size, num_topics)
        self.fclv = nn.Linear(hidden_size, num_topics)
        self.bnmu = nn.BatchNorm1d(num_topics)
        self.bnlv = nn.BatchNorm1d(num_topics)

    def forward(self, inputs):
        h1 = F.softplus(self.fc1(inputs))
        mu = self.bnmu(self.fcmu(h1))
        lv = self.bnlv(self.fclv(h1))
        dist = LogNormal(mu, (0.5 * lv).exp())
        return dist


class Code2Dirichlet(nn.Module):
    """Calculates p(t|z). """

    def __init__(self, code_size, hidden_size, num_topics):
        super().__init__()
        self.fc1 = nn.Linear(code_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_topics)
        self.bn = nn.BatchNorm1d(num_topics)

    def forward(self, inputs):
        h1 = F.softplus(self.fc1(inputs))
        alphas = self.bn(self.fc2(h1)).exp().cpu()
        dist = Dirichlet(alphas)
        return dist

class StandardProir(nn.Module):
    """Calculates p(t)~N(0, I). """
    def __init__(self, num_topics):
        super().__init__()
        batch_size = num_topics.size(0)
        self.mu = torch.full((batch_size, num_topics), 0, dtype=torch.long)
        self.lv = torch.full((batch_size, num_topics), 1, dtype=torch.long)

    def forward(self, inputs):
        dist = LogNormal(self.mu, (0.5 * self.lv).exp())
        return dist


class BilinearFuser(nn.Module):
    """
    Fuse z and t to initialize the hidden state for SeqDecoder.
    z and t are fused with a bilinear layer.
    fuze for specified topic generation
    """

    def __init__(self, code_size, num_topics, hidden_size):
        super().__init__()
        self.bilinear = nn.Bilinear(code_size, num_topics, hidden_size * 2)

    def forward(self, z, t):
        hidden = torch.tanh(self.bilinear(z, t)).unsqueeze(0)
        return [x.contiguous() for x in torch.chunk(hidden, 2, 2)]


class ConcatFuser(nn.Module):
    """
    Fuse z and t to initialize the hidden state for SeqDecoder.
    z and t are fused by applying a linear layer to the concatenation.

    """

    def __init__(self, code_size, num_topics, hidden_size, cell = 'lstm'):
        super().__init__()
        if cell == 'lstm':
            self.fc = nn.Linear(code_size + num_topics, hidden_size * 2)
        elif cell == 'gru':
            self.fc = nn.Linear(code_size + num_topics, hidden_size)

    def forward(self, z, t):
        code = torch.cat([z, t], dim=1)
        hidden = torch.tanh(self.fc(code)).unsqueeze(0)
        return [x.contiguous() for x in torch.chunk(hidden, 2, 2)]


class IdentityFuser(nn.Module):
    """
    Use only z to initialize the hidden state of SeqDecoder.

    """
    def __init__(self, code_size, hidden_size):
        super().__init__()
        self.fc = nn.Linear(code_size, hidden_size * 2)

    def forward(self, z):
        # hidden: [1, hiddensize x 2]
        hidden = torch.tanh(self.fc(z)).unsqueeze(0)
        return [x.contiguous() for x in torch.chunk(hidden, 2, 2)]


class SeqDecoder(nn.Module):
    """
    Decodes into sequences. Calculates p(x|z, t).

    """
    def __init__(self, input_size, hidden_size, dropout, cell='lstm'):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        if cell == 'lstm':
            self.rnn = nn.LSTM(
                input_size, hidden_size, num_layers=1, batch_first=True
            )
        elif cell=='gru':
            self.rnn = nn.GRU(
                input_size, hidden_size, num_layers=1, batch_first=True
            )
        else:
            raise Exception('no such rnn cell')

    def forward(self, inputs, lengths=None, init_hidden=None):
        # inputs size: batch_size x sequence_length x embed_size
        inputs = self.drop(inputs)
        if lengths is not None:
            inputs = pack(inputs, lengths, batch_first=True)
        outputs, hidden = self.rnn(inputs, init_hidden)
        if lengths is not None:
            outputs, _ = unpack(outputs, batch_first=True)
        outputs = self.drop(outputs)
        return outputs, hidden


class BowDecoder(nn.Module):
    """
    Decodes into log-probabilities across the vocabulary.
    Calculates p(x_{bow}|t).

    """

    def __init__(self, vocab_size, num_topics, dropout):
        super().__init__()
        self.fc = nn.Linear(num_topics, vocab_size)
        self.bn = nn.BatchNorm1d(vocab_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        # inputs of size (batch_size x num_topics)
        # returns log probs of each token
        inputs = self.drop(inputs)
        return F.log_softmax(self.bn(self.fc(inputs)), dim=1)


class Discriminator(nn.Module):
    """
    Discriminator to calculate total correlation of z_t
    """
    def __init__(self, num_topics):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_topics, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            # # add begin
            # nn.Linear(1000, 1000),
            # nn.LeakyReLU(0.2, True),
            # nn.Linear(1000, 1000),
            # nn.LeakyReLU(0.2, True),
            # nn.Linear(1000, 1000),
            # nn.LeakyReLU(0.2, True),
            # # add end
            nn.Linear(1000, 2),
        )

    def forward(self, zt):
        return self.net(zt)


class Results:
    """Holds model results. """
    pass


class TopGenVAE(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, hidden_size_t, t_type,
                 code_size, num_topics, dropout, hhf_num_layer, flow, device):
        super().__init__()
        self.t = t_type
        self.lookup = nn.Embedding(vocab_size, embed_size)
        self.seq2bow = Seq2Bow(vocab_size)
        self.encode_seq = SeqEncoder(embed_size, hidden_size, dropout)
        self.encode_bow = BowEncoder(vocab_size, hidden_size_t, dropout)

        self.h2z = Hidden2Normal(hidden_size, code_size)
        self.h2z_hhf = HouseholderFlow(hhf_num_layer, 2 * hidden_size, code_size)
        if flow == 'nmt':
            self.h2z_nmt = RealNVP(hidden_size, code_size)
        # if flow == 'hhf':
        #     self.h2z_hhf = HouseholderFlow(hhf_num_layer, hidden_size, code_size)
        # elif flow =='nmt':
        #     self.h2z_nmt = RealNVP(hidden_size, code_size)

        ########## Apply Normal distribution to model topic ##########
        if t_type == 'normal':
            self.h2t = topic_Hidden2Normal(hidden_size_t, num_topics, device)
            self.z2t = Code2LogNormal(code_size, hidden_size, num_topics)
        ########## Apply Dirichlet distribution to model topic ##########
        elif t_type == 'dirichlet':
            self.h2t = topic_Hidden2Dirichlet(hidden_size_t, num_topics)
            self.z2t = Code2Dirichlet(code_size, hidden_size, num_topics)
        ########## Assume Standard prior to model topic prior ##########
        elif t_type=='standard_normal':
            self.h2t = StandardProir(num_topics)
        else:
            raise Exception('no such distribution for z_t')
        # self.fuse = IdentityFuser(code_size, hidden_size)
        self.fuse = ConcatFuser(code_size, num_topics, hidden_size)
        self.decode_seq = SeqDecoder(embed_size, hidden_size, dropout)
        self.decode_bow = BowDecoder(vocab_size, num_topics, dropout)
        # self.D = Discriminator(num_topics)
        self.D = Discriminator(code_size)
        # output layer
        self.fcout = nn.Linear(hidden_size, vocab_size)
        # self.fcout_from_t = nn.Linear(hidden_size, vocab_size)
        # self.bn_seq_from_zt = nn.BatchNorm1d(vocab_size)
        self.W_t = nn.Linear(num_topics, num_topics, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.flow = flow

    def _encode_z(self, inputs, lengths):
        enc_emb = self.lookup(inputs)
        hn = self.encode_seq(enc_emb, lengths)
        normal_posterior_z, z_start = self.h2z(hn)
        logp = None
        if self.flow == 'hhf':
            posterior_z = self.h2z_hhf(hn, z_start)
        elif self.flow == 'nmt':
            posterior_z, logp = self.h2z_nmt(z_start)
        else:
            # normal distribution
            posterior_z = normal_posterior_z
        return posterior_z, logp, z_start

    def _encode_t(self, inputs, ignore_index):
        # topic latent encoder
        bow_targets = self.seq2bow(inputs, ignore_index)
        h3 = self.encode_bow(bow_targets)
        posterior_t, pos_r = self.h2t(h3)
        return posterior_t, bow_targets, pos_r

    def _encode(self, inputs, lengths, ignore_index):
        pos_z, _, _, = self._encode_z(inputs, lengths)
        pos_t, _, _ = self._encode_t(inputs, ignore_index)
        # print(pos_t.mean, pos_z.mean)
        return pos_z, pos_t

    def _permute_dims(self, z):
        """
        permute latent z_s for tc calculation
        :param z:
        :return:
        """
        assert z.dim() == 2
        B, _ = z.size()
        perm_z = []
        for z_j in z.split(1, 1):
            perm = torch.randperm(B).to(z.device)
            perm_z_j = z_j[perm]
            perm_z.append(perm_z_j)

        return torch.cat(perm_z, 1)

        #######################################################################
        ######## Gumbel-softmax: exp(log(logit_i)+g_i[~GS(0, 1)])/tao #########
        #######################################################################
    def _discriminator_input(self, outputs, tao, bow_embedding=True):
        """
        Calculate x_bar = u^T x W_v : [batch_size, seq_len, embedding_size]
        """
        # outputs [batch_size x seq_len x vocab_size]
        batch_size = outputs.size(0)
        seq_len = outputs.size(1)
        vocab_size = outputs.size(2)
        device = outputs.device
        gumbel_softmax_normal = lambda : -torch.log(-torch.log(torch.rand(vocab_size, device=device))).repeat(batch_size, seq_len, 1)
        g_i = gumbel_softmax_normal()
        u = (outputs + g_i).exp() / tao
        # [batch_size, seq_len, vocab_size]
        u /= torch.sum(u, keepdim=True, dim=2) # torch.sum reduce a dim by default
        ################ Consider BoW embedding as word embedding ################
        if bow_embedding:
            # take mean on max_len dimension
            x_bar = self.encode_bow(torch.mean(u, dim=1))
        else:
            # [batch_size x seq_len x vocab_size x embedding_size]
            W_v = self.lookup(torch.arange(0, vocab_size).repeat(batch_size * seq_len).view(batch_size, seq_len, -1).to(device))
            # print("W_v size", W_v.size())
            # print("u size", u.size())
            # last 2 dims do the mm
            # take mean on second dimension(max_len)
            x_bar = torch.mean(torch.matmul(u.unsqueeze(2), W_v).squeeze(2), dim=1) # [batch_size, embedding_size]
        # print("x_bar size", x_bar.size())
        ############## Word Embedding Size == Hidden Size zt ##############
        posterior_t, _ = self.h2t(x_bar)
        t_discri = posterior_t.rsample().to(device)
        return self.decode_bow(t_discri)

    # def var_loss(self, pos, neg, neg_samples, ignore_index):
    #     """
    #     Only working when self.h2t follows Normal distribution
    #     :param pos:
    #     :param neg:
    #     :param neg_samples:
    #     :param ignore_index:
    #     :return:
    #     """
    #     r = self.encode_bow(self.seq2bow(pos, ignore_index))
    #     _, r = self.h2t(r, True)
    #     pos = self.h2t.encode_var(r)
    #     pos_scores = (pos * r).sum(-1)
    #     pos_scores = pos_scores / torch.norm(r, 2, -1)
    #     pos_scores = pos_scores / torch.norm(pos, 2, -1)
    #     neg = self.encode_bow(self.seq2bow(neg, ignore_index))
    #     _, neg = self.h2t(neg)
    #     neg_scores = (neg * r.repeat(neg_samples, 1)).sum(-1)
    #     neg_scores = neg_scores / torch.norm(r.repeat(neg_samples, 1), 2, -1)
    #     neg_scores = neg_scores / torch.norm(neg, 2, -1)
    #     neg_scores = neg_scores.view(neg_samples, -1)
    #     pos_scores = pos_scores.unsqueeze(0).repeat(neg_samples, 1)
    #     raw_loss = torch.clamp(1 - pos_scores + neg_scores, min=0.).mean(0)
    #     srec_loss = raw_loss.mean()
    #     return srec_loss, raw_loss.sum()

    def inference(self, inputs, neg, neg_samples, lengths, ignore_index):
        # inputs: text input [batch_size x seq_length]
        posterior_z, logp, z_start = self._encode_z(inputs, lengths)
        print("finish z posterior")
        posterior_t, bow_targets, pos_r = self._encode_t(inputs, ignore_index)
        print("finish t posterior")
        # dec_emb: [batch_size x seq_length x embedding_size]
        dec_emb = self.lookup(inputs)
        if self.training:
            # sample from posterior (normal distribution) or specific z (flow) to update latent variable
            if self.flow == 'hhf' or self.flow == 'nmt':
                z = posterior_z
            else:
                z = posterior_z.rsample()
            t = posterior_t.rsample().to(z.device)
            # t = self.softmax(self.W_t(t))
        else:
            if self.flow == 'hhf' or self.flow == 'nmt':
                z = posterior_z
            else:
                z = posterior_z.mean
            t = posterior_t.mean.to(z.device)
            # t = self.softmax(self.W_t(t))
        bow_outputs = self.decode_bow(t)
        hidden = self.fuse(z, t)
        # [batch_size x seq_len x hidden_size]
        outputs, _ = self.decode_seq(dec_emb, lengths, hidden)

        # inference_zt = self.seq2hidden_t(inputs)
        # [batch_size x vocab_size]
        # seq_from_zt = self.bn_seq_from_zt(self.fcout_from_t(zt_hidden))
        #######################################################################
        ####################### Discriminator Calculation #####################
        #######################################################################
        seq_outputs = self.fcout(outputs)  # [batch_size x seq_len x vocab_size]
        if self.training:
            tao = torch.tensor(0.02, device=z.device)
        else:
            tao = torch.tensor(1, device=z.device)
        discri_log = self._discriminator_input(seq_outputs, tao)

        ####################### z_t TC Calculation #####################
        # D_zt = self.D(t)
        # zt_permute = self._permute_dims(t)
        # D_zt_permute = self.D(zt_permute)
        ####################### z_s TC Calculation #####################
        D_zt = self.D(z)
        zt_permute = self._permute_dims(z)
        D_zt_permute = self.D(zt_permute)
        print("start cp loss")

        r = self.encode_bow(self.seq2bow(inputs, ignore_index))
        _, r = self.h2t(r, True)
        pos = pos_r
        pos_scores = (pos * r).sum(-1)
        pos_scores = pos_scores / torch.norm(r, 2, -1)
        pos_scores = pos_scores / torch.norm(pos, 2, -1)
        neg = self.encode_bow(self.seq2bow(neg, ignore_index))
        print("finish encodebow of neg")
        _, neg = self.h2t(neg)
        neg_scores = (neg * r.repeat(neg_samples, 1)).sum(-1)
        neg_scores = neg_scores / torch.norm(r.repeat(neg_samples, 1), 2, -1)
        neg_scores = neg_scores / torch.norm(neg, 2, -1)
        neg_scores = neg_scores.view(neg_samples, -1)
        pos_scores = pos_scores.unsqueeze(0).repeat(neg_samples, 1)

        results = Results()
        results.z = z
        results.z_start = z_start
        # results.z_mu = z_mu
        # results.z_lv = z_lv
        results.t = t
        results.bow_targets = bow_targets
        results.seq_outputs = seq_outputs
        results.bow_outputs = bow_outputs
        """
        p(z): prior z     | from sequential decoder | follow normal isotropic gaussian N(0, I)
        q(z): posterior z | from sequential encoder | modeled by householder transformation, considered as a 
                          |                         | **complex distribution**, updates in incorporating topic way
        p(t): prior t     | from topic decoder      | follow dirichlet distribution, condition on posterior z q(z)
        q(t): posterior t | from topic encoder      | follow dirichlet distribution
        """
        results.posterior_z = posterior_z
        results.prior_t = self.z2t(z)
        results.posterior_t = posterior_t
        results.bow_outputs_probs = bow_outputs.exp()
        results.discri_log = discri_log
        results.D_zt = D_zt
        results.D_zt_permute = D_zt_permute
        results.nmt_logp = logp
        if self.t == 'normal':
            print("start reg")
            results.reg_loss = self.h2t.orthogonal_regularizer()
            print("finish reg")
            results.pos_scores = pos_scores
            results.neg_scores = neg_scores

        return results

    def forward(self, inputs, lengths, ignore_index):
        # inputs: text input [batch_size x seq_length]
        posterior_z, logp, z_start = self._encode_z(inputs, lengths)
        posterior_t, bow_targets, pos_r = self._encode_t(inputs, ignore_index)
        # dec_emb: [batch_size x seq_length x embedding_size]
        dec_emb = self.lookup(inputs)
        if self.training:
            # sample from posterior (normal distribution) or specific z (flow) to update latent variable
            if self.flow == 'hhf' or self.flow == 'nmt':
                z = posterior_z
            else:
                z = posterior_z.rsample()
            t = posterior_t.rsample().to(z.device)
        else:
            if self.flow == 'hhf' or self.flow == 'nmt':
                z = posterior_z
            else:
                z = posterior_z.mean
            t = posterior_t.mean.to(z.device)
        bow_outputs = self.decode_bow(t)
        hidden = self.fuse(z, t)
        # [batch_size x seq_len x hidden_size]
        outputs, _ = self.decode_seq(dec_emb, lengths, hidden)

        # inference_zt = self.seq2hidden_t(inputs)
        # [batch_size x vocab_size]
        # seq_from_zt = self.bn_seq_from_zt(self.fcout_from_t(zt_hidden))
        #######################################################################
        ####################### Discriminator Calculation #####################
        #######################################################################
        seq_outputs = self.fcout(outputs) # [batch_size x seq_len x vocab_size]
        if self.training:
            tao = torch.tensor(0.02, device=z.device)
        else:
            tao = torch.tensor(1, device=z.device)
        discri_log = self._discriminator_input(seq_outputs, tao)

        ####################### z_t TC Calculation #####################
        # D_zt = self.D(t)
        # zt_permute = self._permute_dims(t)
        # D_zt_permute = self.D(zt_permute)
        ####################### z_s TC Calculation #####################
        D_zt = self.D(z)
        zt_permute = self._permute_dims(z)
        D_zt_permute = self.D(zt_permute)

        results = Results()
        results.z = z
        results.z_start = z_start
        # results.z_mu = z_mu
        # results.z_lv = z_lv
        results.t = t
        results.bow_targets = bow_targets
        results.seq_outputs = seq_outputs
        results.bow_outputs = bow_outputs
        """
        p(z): prior z     | from sequential decoder | follow normal isotropic gaussian N(0, I)
        q(z): posterior z | from sequential encoder | modeled by householder transformation, considered as a 
                          |                         | **complex distribution**, updates in incorporating topic way
        p(t): prior t     | from topic decoder      | follow dirichlet distribution, condition on posterior z q(z)
        q(t): posterior t | from topic encoder      | follow dirichlet distribution
        """
        results.posterior_z = posterior_z
        results.prior_t = self.z2t(z)
        results.posterior_t = posterior_t
        results.bow_outputs_probs = bow_outputs.exp()
        results.discri_log = discri_log
        results.D_zt = D_zt
        results.D_zt_permute = D_zt_permute
        results.nmt_logp = logp
        # results.inference_zt = inference_zt
        # [batch_size x topic_num]
        # results.topic_mu = topic_mu
        # results.seq_from_zt = seq_from_zt

        return results

    def generate(self, z, t, max_length, sos_id, sample_sign=False):
        batch_size = z.size(0)
        hidden = self.fuse(z, t)
        generated = torch.zeros((batch_size, max_length), dtype=torch.long, device=z.device)
        dec_inputs = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=z.device)
        for k in range(max_length):
            dec_emb = self.lookup(dec_inputs)
            outputs, hidden = self.decode_seq(dec_emb, init_hidden=hidden)
            outputs = self.fcout(outputs)
            if sample_sign is True:
                dec_inputs = torch.multinomial(outputs.view(-1, outputs.size(2)), 1) # categorical sample word generation
            else:
                dec_inputs = outputs.max(2)[1] # index of max prob word
            generated[:, k] = dec_inputs[:, 0].clone()
        return generated

    def beamsearch_generation(self, num_samples, max_length, sos_id, eos_id, device, beam_width=8, z=None, t=None):
        code_size = self.z2t.fc1.in_features
        if z is None:
            # p(z)~ N(0, I)
            z = torch.randn(num_samples, code_size, device=device)
            prior_t = self.z2t(z)
            t = prior_t.sample().to(device)
        else:
            z = z
            t = t
        batch_size = num_samples
        hidden = self.fuse(z, t)
        # n_best_list = batch_beam_search_decoding(self.decode_seq, hidden,
        #                                          beam_width, n_best, batch_size,
        #                                          sos_id, self.fcout, self.lookup,
        #                                          eos_id, max_length, device)
        n_best_list = beam_decode(self.decode_seq, batch_size, hidden, device, sos_id, eos_id, self.fcout, self.lookup, beam_width)
        return n_best_list


    def reconstruct(self, inputs, lengths, ignore_index, max_length, sos_id, eos_id, flow, beam_search=False,
                    fix_z=True, fix_t=True):
        """
        reconstruct from input sentences
        :param inputs:
        :param lengths:
        :param pad_id:
        :param max_length:
        :param sos_id:
        :param fix_z:
        :param fix_t:
        :return:
        """
        posterior_z, posterior_t = self._encode(inputs, lengths, ignore_index)
        if flow == 'hhf' or 'nmt':
            if fix_z:
                z = posterior_z
            else:
                raise Exception('flow posterior has no sample method')
        else:
            if fix_z:
                z = posterior_z.mean
            else:
                z = posterior_z.sample()
        if fix_t:
            t = posterior_t.mean.to(z.device)
        else:
            t = posterior_t.sample().to(z.device)
        if beam_search:
            return beam_decode(self.decode_seq, z.size(0), self.fuse(z, t), z.device,
                               sos_id, eos_id, self.fcout, self.lookup, beam_width=8)
        else:
            return self.generate(z, t, max_length, sos_id)

    def sample(self, num_samples, max_length, sos_id, device):
        """
        Randomly sample latent code to sample texts.
        """
        code_size = self.z2t.fc1.in_features
        # p(z)~ N(0, I)
        z = torch.randn(num_samples, code_size, device=device)
        prior_t = self.z2t(z)
        t = prior_t.sample().to(device)
        return self.generate(z, t, max_length, sos_id)

    def get_topics(self, inputs, ignore_index):
        posterior_t, _, _ = self._encode_t(inputs, ignore_index)
        t = posterior_t.mean.to(inputs.device)
        return t / t.sum(1, keepdim=True)

    def _interpolate_init(self, pairs, i, n):
        # combine 2 sentence features in a pair ?
        x1, x2 = [x.clone() for x in pairs]
        # x1, x2 = [x for x in pairs]
        return x1 * (n - 1 - i) / (n - 1) + x2 * i / (n - 1)

    def _interpolate(self, pairs, i, n):
        # combine 2 sentence features in a pair ?
        x_final = torch.tensor([]).to(pairs[0].device)
        for ii in pairs:
            x1, x2 = [x.clone() for x in ii]
            interpreted_num = x1 * (n - 1 - i) / (n - 1) + x2 * i / (n - 1)
            x_final = torch.cat([x_final, interpreted_num.unsqueeze(0)], dim=0)
        # x1, x2 = [x for x in pairs]
        return x_final

    def interpolate(self, input_pairs, length_pairs, ignore_index, max_length, sos_id, flow, num_pts=4, fix_z=False, fix_t=False):
        """
        interpolation of generated sentence smooth trasition between 2 sententces
        :param input_pairs:
        :param length_pairs:
        :param pad_id:
        :param max_length:
        :param sos_id:
        :param num_pts:
        :return:
        """
        z_pairs = []
        t_pairs = []
        for inputs, lengths in zip(input_pairs, length_pairs):
            posterior_z, posterior_t = self._encode(inputs, lengths, ignore_index)
            z = posterior_z if flow ==('hhf' or 'nmt') else posterior_z.mean
            t = posterior_t.mean.to(z.device)
            z_pairs.append(z)
            t_pairs.append(t)
        generated = []
        for i in range(num_pts + 2):
            # print(len(z_pairs), z_pairs)
            if fix_z:
                z = z[0, :].unsqueeze(0)
            else:
                z = self._interpolate(z_pairs, i, num_pts + 2)
            if fix_t:
                t = t[0, :].unsqueeze(0)
            else:
                t = self._interpolate(t_pairs, i, num_pts + 2)
            generated.append(self.generate(z, t, max_length, sos_id))
        return generated





