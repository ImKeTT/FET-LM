import sys
import json
import re
import os
import time
import csv
#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: train.py
@author: ImKe at 2021/3/21
@feature: #Enter features here
@scenario: #Enter scenarios here
"""
import argparse
import time
import random
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import re
import numpy as np
import datetime

from topic_model import TopGenVAE
from data_loader import Corpus, get_iterator, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
from loss import seq_recon_loss, bow_recon_loss, discriminator_loss
from logger import Logger
from torch.nn.utils import clip_grad_norm

parser = argparse.ArgumentParser(description='Text VAE')
parser.add_argument('--data', type=str, default='./data/ptb',
                    help="location of the data folder")
parser.add_argument('--splist', type=str, default='./data/stopwords.txt',
                    help="location of the stopwords file")
parser.add_argument('--max_vocab', type=int, default=20000,
                    help="maximum vocabulary size for the input")
parser.add_argument('--max_length', type=int, default=500,
                    help="maximum sequence length for the input")
parser.add_argument('--embed_size', type=int, default=200,
                    help="size of the word embedding")
parser.add_argument('--label_embed_size', type=int, default=8,
                    help="size of the label embedding")
parser.add_argument('--hidden_size', type=int, default=300,
                    help="number of hidden units for RNN")
parser.add_argument('--hidden_size_t', type=int, default=200,
                    help="number of hidden units for RNN")
parser.add_argument('--code_size', type=int, default=32,
                    help="latent code dimension")
parser.add_argument('--num_topics', type=int, default=20, # yelp 6topics, yahoo 10 topics
                    help="number of topics")
parser.add_argument('--epochs', type=int, default=80,
                    help="maximum training epochs")
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help="batch size")
parser.add_argument('--dropout', type=float, default=0.2,
                    help="dropout applied to layers (0 = no dropout)")
parser.add_argument('--alpha', type=float, default=1.0,
                    help="weight of the mutual information term")
parser.add_argument('--beta', type=float, default=0.5,
                    help="topic latent weight")
parser.add_argument('--gamma', type=float, default=0.1,
                    help="weight of the discriminator")
parser.add_argument('--lr', type=float, default=1e-4,
                    help="learning rate")
parser.add_argument('--wd', type=float, default=1e-5,
                    help="weight decay used for regularization")
parser.add_argument('--tc_weight', type=float, default=1.0,
                    help="total correlation weight")
parser.add_argument('--clip', type=float, default=5.0,
                    help="max clip norm of gradient clip")
parser.add_argument('--flow_type', type=str, default=None,
                    help="Choose hhf or nmt flow")
parser.add_argument('--t_type', type=str, default='dirichlet',
                    help="Choose normal or dirichlet to model topic latent variable")
parser.add_argument('--flow_num_layer', type=int, default=10,
                    help="Householder flow layer number")
parser.add_argument('--epoch_size', type=int, default=2000,
                    help="number of training steps in an epoch")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed")
parser.add_argument('--kla', type=str, default='cyc',
                    help="use kl annealing cyc or mono")
parser.add_argument('--nocuda', action='store_true',
                    help="do not use CUDA")
args = parser.parse_known_args()[0]

torch.manual_seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def say_sents(corpus, tex, eos_id):
    final_sent = []
    for sent in tex:
        sent_list = []
        for w in sent:
            if w != eos_id:
                sent_list.append(corpus.idx2word[w])
            else:
                break
        final_sent.append(sent_list)
    return final_sent

if __name__=='__main__':
    # # with open('./data/yelp/yelp_academic_dataset_tip.json', 'r') as f:
    # #     dataset = f.read().split('\n')
    # # print(dataset[:10])
    # name = 'train'
    # filename = f'./data/yahoo/{name}.csv'
    # datalist = []
    # with open(filename) as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         datalist.append(row)
    # with open(f'./data/yahoo/{name}.txt', 'w') as f:
    #     for ins in datalist:
    #         f.write(str(ins[0]) + '\t' + str(ins[1]) + '\t' + str(ins[2]) + '\t' + str(ins[3]) + '\n')
    # f.close()

    print("Loading data")
    dataset = args.data.rstrip('/').split('/')[-1]
    if dataset in ['yahoo', 'yelp']:
        with_label = True
    else:
        with_label = False
    corpus = Corpus(
        args.data, max_vocab_size=args.max_vocab,
        max_length=args.max_length, with_label=with_label
    )
    pad_id = corpus.word2idx[PAD_TOKEN]
    sos_id = corpus.word2idx[SOS_TOKEN]
    eos_id = corpus.word2idx[EOS_TOKEN]
    vocab_size = len(corpus.word2idx)
    print("\ttraining data size: ", len(corpus.train))
    print("\tvocabulary size: ", vocab_size)
    print("Constructing model")
    print(args)
    device = 'cuda'  # torch.device('cpu' if args.nocuda else 'cuda')
    flow = args.flow_type
    model = TopGenVAE(
        vocab_size, args.embed_size, args.hidden_size, args.hidden_size_t, args.t_type, args.code_size,
        args.num_topics, args.dropout, args.flow_num_layer, flow
    ).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # best_loss = None
    print('model structure:')
    print(model)
    print('Loading checkpoints')
    # ckpt_file = './ckpt/ptb/emb200.hid300.z32.t20_dirichlet..wd1e-05.kla.tc1.0.beta1.0.hhf.flow.ptb.80epoch.date4.2.pt'
    ckpt_file = './ckpt/ptb/emb200.hid300.z32.t20_dirichlet..wd1e-05.kla_cyc.tc1.0.beta0.5.hhf.flow.ptb.81epoch.date4.20.disw0.1.pt'
    with open(ckpt_file, 'rb') as f:
        ckpt = torch.load(f)
        model.load_state_dict(ckpt['model'])
    print('Finish loading ckpt')
    model.eval()
    """
    # save generated sentences
    num_samples = 10
    max_length = 30
    sents = model.beamsearch_generation(num_samples, max_length, sos_id, eos_id, device)
    print(len(sents), len(sents[0]), len(sents[0][0]))
    # sents = [int(sents[0][0][i][0][0])for i in range(len(sents[0][0]))]
    sents = [[[int(sents[k][j][i][0][0]) for i in range(len(sents[k][j]))] for j in range(len(sents[k]))] for k in range(len(sents))]
    word_sent = [[[corpus.idx2word[w] for w in sent] for sent in ex] for ex in sents]
    # word_sent = [[[corpus.idx2word[w] for w in sent] for sent in ex] for ex in sents]
    with open(f'./results/test_ptb_{num_samples}.txt', 'w', encoding='utf8') as f:
        for sent in word_sent:
            f.write(" ".join(str(i) for i in sent[0][1 : -2]) + '\n')
    print(f'Finish writing {num_samples} sentences in file')

    # word_sent = [[corpus.idx2word[w] for w in sents]]
    # for sent in word_sent:
    #     print(" ".join(str(i) for i in sent))
    """
    valid_iter = get_iterator(corpus.test, 2, False, device)
    tex0 = []
    length0 = []
    for i, batch in enumerate(valid_iter):
        tex, length = batch.text
        tex0.append(tex)
        length0.append(length)
        break
    tex_inter = model.interpolate(tex0[0], length0[0], pad_id, args.max_length, sos_id, flow, num_pts=4)
    word_sent = [[corpus.idx2word[w] for w in sent] for sent in tex]
    print('Original text:')
    for sent in word_sent:
        print(" ".join(str(i) for i in sent))
    tex_rec = model.reconstruct(tex, length, pad_id, args.max_length, sos_id, flow)
    # word_sent0 = [[corpus.idx2word[w] for w in sent] for sent in tex_rec]
    word_sent0 = say_sents(corpus, tex_rec, eos_id)
    for i, tex_inter_sent in enumerate(tex_inter):
        print(f'No.{i} interpreted sent:')
        word_sent1 = say_sents(corpus, tex_inter_sent, eos_id)
        # word_sent1 = [[corpus.idx2word[w] for w in sent] for sent in tex_inter_sent]
        for sent in word_sent1:
            print(" ".join(str(i) for i in sent))
    print('RECONSTRUCTING...')
    for sent in word_sent0:
        print(" ".join(str(i) for i in sent))