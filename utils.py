#!/usr/bin/env python
#-*- coding: utf-8 -*-

import torch
import copy
import torch.nn as nn
import operator
from heapq import heappush, heappop
from queue import PriorityQueue
from typing import Tuple, List


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True,
                            bidirectional=True)
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=True)
        self.fc_lstm = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection


    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size).to(self.device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc_lstm(out[:, -1, :])
        return out

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


def batch_beam_search_decoding(decoder, hidden, beam_width, n_best, bs,
                               sos_token, fout_layer, lookup_layer, eos_token, max_length, device):
    """Batch Beam Seach Decoding for RNN
    Args:
        decoder: An RNN decoder model
        enc_outs: A sequence of encoded input. (T, bs, 2H). 2H for bidirectional
        enc_last_h: (bs, H)
        beam_width: Beam search width
        n_best: The number of output sequences for each input
        bs: batch_size
    Returns:
        n_best_list: Decoded N-best results. (bs, T)
    """

    assert beam_width >= n_best

    n_best_list = []

    # Get last encoder hidden state
    decoder_hidden = hidden # (bs, H)
    # for idx in range(bs): #lstm case
    #     decoder_hidden = (hidden[0][:, idx, :].unsqueeze(0), hidden[1][:, idx, :].unsqueeze(0))

    # Prepare first token for decoder
    decoder_input = torch.tensor([sos_token]).repeat(1, bs).long().to(device) # (1, bs)

    # Number of sentence to generate
    end_nodes_list = [[] for _ in range(bs)]

    # whole beam search node graph
    nodes = [[] for _ in range(bs)]

    # Start the queue
    # print(len(decoder_hidden), decoder_hidden[0].size())
    for bid in range(bs):
        # gru case
        # node = BeamSearchNode(hiddenstate=decoder_hidden[bid],
        #                       previousNode=None,
        #                       wordId=decoder_input[:, bid], logProb=0, length=1)
        # starting node : for lstm case
        node = BeamSearchNode(hiddenstate=[decoder_hidden[0][:, bid, :], decoder_hidden[1][:, bid, :]],
                              previousNode=None,
                              wordId=decoder_input[:, bid], logProb=0, length=1)
        heappush(nodes[bid], (-node.eval(), id(node), node))


    # Start beam search
    fin_nodes = set()
    history = [None for _ in range(bs)]
    n_dec_steps_list = [0 for _ in range(bs)]
    while len(fin_nodes) < bs:
        # Fetch the best node
        decoder_input, decoder_hidden = [], []
        for bid in range(bs):
            if bid not in fin_nodes and n_dec_steps_list[bid] > max_length:
                fin_nodes.add(bid)

            if bid in fin_nodes:
                score, n = history[bid] # dummy for data consistency
            else:
                score, _, n = heappop(nodes[bid])
                if n.wid.item() == eos_token and n.prev_node is not None:
                    end_nodes_list[bid].append((score, id(n), n))
                    # If we reached maximum # of sentences required
                    if len(end_nodes_list[bid]) >= n_best:
                        fin_nodes.add(bid)
                history[bid] = (score, n)
            decoder_input.append(n.wid)
            decoder_hidden.append(n.h)

        decoder_input = torch.cat(decoder_input).unsqueeze(1).to(device) # (bs)
        tensor0 = torch.tensor([[]]).to(device)
        tensor1 = torch.tensor([[]]).to(device)
        for tt in decoder_hidden:
            tensor0 = torch.cat([tensor0, tt[0]], dim=1)
            tensor1 = torch.cat([tensor1, tt[1]], dim=1)
        decoder_hidden = [tensor0.view(1, bs, -1), tensor1.view(1, bs, -1)]
        # decoder_hidden = torch.stack(decoder_hidden, 0).to(device) # (bs, H)

        # Decode for one step using decoder
        decoder_output, decoder_hidden = decoder(lookup_layer(decoder_input), init_hidden=decoder_hidden) # (bs, V), (bs, H)

        # Get top-k from this decoded result
        topk_log_prob, topk_indexes = torch.topk(fout_layer(decoder_output), beam_width) # (bs, bw), (bs, bw)
        # print(topk_indexes.size())
        # Then, register new top-k nodes
        for bid in range(bs):
            if bid in fin_nodes:
                continue
            score, n = history[bid]
            if n.wid.item() == eos_token and n.prev_node is not None:
                continue
            for new_k in range(beam_width):
                # for GRU case
                # decoded_t = topk_indexes[bid][0][new_k].view(1)  # (1)
                # logp = topk_log_prob[bid][0][new_k].item()  # float log probability val
                # for LSTM case if GRU then
                decoded_t = topk_indexes[bid][0][new_k].view(1) # (1)
                logp = topk_log_prob[bid][0][new_k].item() # float log probability val

                # lstm case
                node = BeamSearchNode(hiddenstate=[decoder_hidden[0][:, bid, :], decoder_hidden[1][:, bid, :]],
                                      previousNode=n,
                                      wordId=decoded_t,
                                      logProb=n.logp+logp,
                                      length=n.leng+1)
                # gru case
                # node = BeamSearchNode(hiddenstate=decoder_hidden[bid],
                #                       previousNode=n,
                #                       wordId=decoded_t,
                #                       logProb=n.logp + logp,
                #                       length=n.leng + 1)
                heappush(nodes[bid], (-node.eval(), id(node), node))
            n_dec_steps_list[bid] += beam_width

    # Construct sequences from end_nodes
    # if there are no end_nodes, retrieve best nodes (they are probably truncated)
    for bid in range(bs):
        if len(end_nodes_list[bid]) == 0:
            end_nodes_list[bid] = [heappop(nodes[bid]) for _ in range(beam_width)]

        n_best_seq_list = []
        for score, _id, n in sorted(end_nodes_list[bid], key=lambda x: x[0]):
            sequence = [n.wid.item()]
            while n.prevNode is not None:
                n = n.prevNode
                sequence.append(n.wid.item())
            sequence = sequence[::-1] # reverse

            n_best_seq_list.append(sequence)

        n_best_list.append(copy.copy(n_best_seq_list))

    return n_best_list



def beam_decode(decoder, bs, decoder_hiddens, device, sos_id, eos_id, fout_layer, lookup_layer, beam_width, topk=1):
    '''
    :param decoder: rnn decoder for word generation
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    beam_width = beam_width
    topk = topk  # how many sentence do you want to generate
    decoded_batch = []

    # decoding goes sentence by sentence
    for idx in range(bs):
        if isinstance(decoder_hiddens, list):  # LSTM case
            decoder_hidden = (decoder_hiddens[0][:,idx, :].unsqueeze(0),decoder_hiddens[1][:,idx, :].unsqueeze(0))
        else:
            decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)

        # Start with the start of the sentence token
        decoder_input = torch.LongTensor([[sos_id]]).to(device)
        # decoder_input = torch.full((bs, 1), sos_id, dtype=torch.long, device=device)


        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 2000: break

            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.wid
            decoder_hidden = n.h

            if n.wid.item() == eos_id and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            # print(len(decoder_hidden), decoder_hidden[0].size())
            decoder_output, decoder_hidden = decoder(lookup_layer(decoder_input), init_hidden=list(decoder_hidden))

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(fout_layer(decoder_output), beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                # todo: batch operation
                # generate one sentence per time ----> slow
                decoded_t = indexes[0][0][new_k].view(1, -1)
                log_p = log_prob[0][0][new_k].item()

                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wid)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wid)

            utterance = utterance[::-1]
            utterances.append(utterance)

        decoded_batch.append(utterances)

    return decoded_batch


def beam_search_decode_v3(mode, z1, z2=None, K=5, max_t=20):
    decoded_batch = []
    if z2 is not None:
        z = torch.cat([z1, z2], -1)
    else:
        z = z1
    batch_size, nz = z.size()

    c_init = mode.trans_linear(z).unsqueeze(0)
    h_init = torch.tanh(c_init)

    for idx in range(batch_size):
        decoder_input = torch.tensor([[mode.vocab["<s>"]]], dtype=torch.long,
                                     device=mode.device)
        decoder_hidden = (h_init[:, idx, :].unsqueeze(1), c_init[:, idx, :].unsqueeze(1))
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0.1, 1)
        live_hypotheses = [node]

        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < K and t < max_t:
            t += 1

            decoder_input = torch.cat([node.wordid for node in live_hypotheses], dim=1)

            decoder_hidden_h = torch.cat([node.h[0] for node in live_hypotheses], dim=1)
            decoder_hidden_c = torch.cat([node.h[1] for node in live_hypotheses], dim=1)

            decoder_hidden = (decoder_hidden_h, decoder_hidden_c)

            word_embed = mode.embed(decoder_input)
            word_embed = torch.cat((word_embed, z[idx].view(1, 1, -1).expand(
                1, len(live_hypotheses), nz)), dim=-1)

            output, decoder_hidden = mode.lstm(word_embed, decoder_hidden)

            output_logits = mode.pred_linear(output)
            decoder_output = F.log_softmax(output_logits, dim=-1)

            prev_logp = torch.tensor([node.logp for node in live_hypotheses],
                                     dtype=torch.float, device=mode.device)
            decoder_output = decoder_output + prev_logp.view(1, len(live_hypotheses), 1)

            decoder_output = decoder_output.view(-1)

            log_prob, indexes = torch.topk(decoder_output, K - len(completed_hypotheses))

            live_ids = indexes // len(mode.vocab)
            word_ids = indexes % len(mode.vocab)

            live_hypotheses_new = []
            for live_id, word_id, log_prob_ in zip(live_ids, word_ids, log_prob):
                node = BeamSearchNode((
                    decoder_hidden[0][:, live_id, :].unsqueeze(1),
                    decoder_hidden[1][:, live_id, :].unsqueeze(1)),
                    live_hypotheses[live_id], word_id.view(1, 1), log_prob_, t)

                if word_id.item() == mode.vocab["<EOS>"]:
                    completed_hypotheses.append(node)
                else:
                    live_hypotheses_new.append(node)

            live_hypotheses = live_hypotheses_new

            if len(completed_hypotheses) == K:
                break

        for live in live_hypotheses:
            completed_hypotheses.append(live)

        utterances = []
        for n in sorted(completed_hypotheses, key=lambda node: node.logp, reverse=True):
            utterance = []
            utterance.append(mode.vocab.id2word(n.wordid.item()))
            while n.prevNode is not None:
                n = n.prevNode
                utterance.append(mode.vocab.id2word(n.wordid.item()))

            utterance = utterance[::-1]
            utterances.append(utterance)

            break

        decoded_batch.append(utterances[0])

    return decoded_batch


def greedy_decode(decoder, decoder_hidden, encoder_outputs, target_tensor, sos_id, device, MAX_LENGTH):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    batch_size, seq_len = target_tensor.size()
    decoded_batch = torch.zeros((batch_size, MAX_LENGTH))
    decoder_input = torch.LongTensor([[sos_id] for _ in range(batch_size)], device=device)

    for t in range(MAX_LENGTH):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

        topv, topi = decoder_output.data.topk(1)  # get candidates
        topi = topi.view(-1)
        decoded_batch[:, t] = topi

        decoder_input = topi.detach().view(-1, 1)

    return decoded_batch


def reduce_mean(tensor, dim=None, keepdim=False, out=None):
    """
    Returns the mean value of each row of the input tensor in the given dimension dim.

    Support multi-dim mean

    :param tensor: the input tensor
    :type tensor: torch.Tensor
    :param dim: the dimension to reduce
    :type dim: int or list[int]
    :param keepdim: whether the output tensor has dim retained or not
    :type keepdim: bool
    :param out: the output tensor
    :type out: torch.Tensor
    :return: mean result
    :rtype: torch.Tensor
    """
    # mean all dims
    if dim is None:
        return torch.mean(tensor)
    # prepare dim
    if isinstance(dim, int):
        dim = [dim]
    dim = sorted(dim)
    # get mean dim by dim
    for d in dim:
        tensor = tensor.mean(dim=d, keepdim=True)
    # squeeze reduced dimensions if not keeping dim
    if not keepdim:
        for cnt, d in enumerate(dim):
            tensor.squeeze_(d - cnt)
    if out is not None:
        out.copy_(tensor)
    return tensor


def reduce_sum(tensor, dim=None, keepdim=False, out=None):
    """
    Returns the sum of all elements in the input tensor.

    Support multi-dim sum

    :param tensor: the input tensor
    :type tensor: torch.Tensor
    :param dim: the dimension to reduce
    :type dim: int or list[int]
    :param keepdim: whether the output tensor has dim retained or not
    :type keepdim: bool
    :param out: the output tensor
    :type out: torch.Tensor
    :return: sum result
    :rtype: torch.Tensor
    """
    # summarize all dims
    if dim is None:
        return torch.sum(tensor)
    # prepare dim
    if isinstance(dim, int):
        dim = [dim]
    dim = sorted(dim)
    # get summary dim by dim
    for d in dim:
        tensor = tensor.sum(dim=d, keepdim=True)
    # squeeze reduced dimensions if not keeping dim
    if not keepdim:
        for cnt, d in enumerate(dim):
            tensor.squeeze_(d - cnt)
    if out is not None:
        out.copy_(tensor)
    return tensor


def tensor_equal(a, b, eps=1e-6):
    """
    Compare two tensors

    :param a: input tensor a
    :type a: torch.Tensor
    :param b: input tensor b
    :type b: torch.Tensor
    :param eps: epsilon
    :type eps: float
    :return: whether two tensors are equal
    :rtype: bool
    """
    if a.shape != b.shape:
        return False

    return 0 <= float(torch.max(torch.abs(a - b))) <= eps


def split_channel(tensor, split_type='simple'):
    """
    Split channels of tensor

    :param tensor: input tensor
    :type tensor: torch.Tensor
    :param split_type: type of splitting
    :type split_type: str
    :return: split tensor
    :rtype: tuple(torch.Tensor, torch.Tensor)
    """
    assert len(tensor.shape) == 4
    assert split_type in ['simple', 'cross']

    nc = tensor.shape[1]
    if split_type == 'simple':
        return tensor[:, :nc // 2, ...], tensor[:, nc // 2:, ...]
    elif split_type == 'cross':
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


def cat_channel(a, b):
    """
    Concatenates channels of tensors

    :param a: input tensor a
    :type a: torch.Tensor
    :param b: input tensor b
    :type b: torch.Tensor
    :return: concatenated tensor
    :rtype: torch.Tensor
    """
    return torch.cat((a, b), dim=1)


def count_pixels(tensor):
    """
    Count number of pixels in given tensor

    :param tensor: input tensor
    :type tensor: torch.Tensor
    :return: number of pixels
    :rtype: int
    """
    assert len(tensor.shape) == 4
    return int(tensor.shape[2] * tensor.shape[3])


def onehot(y, num_classes):
    """
    Generate one-hot vector

    :param y: ground truth labels
    :type y: torch.Tensor
    :param num_classes: number os classes
    :type num_classes: int
    :return: one-hot vector generated from labels
    :rtype: torch.Tensor
    """
    assert len(y.shape) in [1, 2], "Label y should be 1D or 2D vector"
    y_onehot = torch.zeros(y.shape[0], num_classes, device=y.device)
    if len(y.shape) == 1:
        y_onehot = y_onehot.scatter_(1, y.unsqueeze(-1), 1)
    else:
        y_onehot = y_onehot.scatter_(1, y, 1)
    return y_onehot

'''
def squeeze(x: torch.Tensor, mask: torch.Tensor, factor: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x: Tensor
            input tensor [batch, length, features]
        mask: Tensor
            mask tensor [batch, length]
        factor: int
            squeeze factor (default 2)
    Returns: Tensor1, Tensor2
        squeezed x [batch, length // factor, factor * features]
        squeezed mask [batch, length // factor]
    """
    assert factor >= 1
    if factor == 1:
        return x

    batch, length, features = x.size()
    assert length % factor == 0
    # [batch, length // factor, factor * features]
    x = x.contiguous().view(batch, length // factor, factor * features)
    mask = mask.view(batch, length // factor, factor).sum(dim=2).clamp(max=1.0)
    return x, mask


def unsqueeze(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """
    Args:
        x: Tensor
            input tensor [batch, length, features]
        factor: int
            unsqueeze factor (default 2)
    Returns: Tensor
        squeezed tensor [batch, length * 2, features // 2]
    """
    assert factor >= 1
    if factor == 1:
        return x

    batch, length, features = x.size()
    assert features % factor == 0
    # [batch, length * factor, features // factor]
    x = x.view(batch, length * factor, features // factor)
    return x


def split(x: torch.Tensor, z1_features) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x: Tensor
            input tensor [batch, length, features]
        z1_features: int
            the number of features of z1
    Returns: Tensor, Tensor
        split tensors [batch, length, z1_features], [batch, length, features-z1_features]
    """
    z1 = x[:, :, :z1_features]
    z2 = x[:, :, z1_features:]
    return z1, z2


def unsplit(xs: List[torch.Tensor]) -> torch.Tensor:
    """
    Args:
        xs: List[Tensor]
            tensors to be combined
    Returns: Tensor
        combined tensor
    """
    return torch.cat(xs, dim=2)


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    mask = tensor.ne(padding_idx).long()
    return torch.cumsum(mask, dim=1) * mask
'''
