#!/usr/bin/env python
#-*- coding: utf-8 -*-
import argparse
import time
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import datetime

from topic_model import TopGenVAE
from data_loader import Corpus, get_iterator, PAD_TOKEN, SOS_TOKEN, get_neg_batch
from loss import seq_recon_loss, bow_recon_loss, discriminator_loss
from loss import normal_kld, cal_tc, hhf_kld, nmt_kld, mmd, mmd_v2
from logger import Logger
from torch.nn.utils import clip_grad_norm


parser = argparse.ArgumentParser(description='Text VAE')
parser.add_argument('--data', type=str, default='apnews',
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
parser.add_argument('--num_topics', type=int, default=50, # vary from [10, 30, 50]
                    help="number of topics")
parser.add_argument('--min_freq', type=int, default=2, # minimum frequency for topic model
                    help="min frequency for corpus")
parser.add_argument('--epochs', type=int, default=80,
                    help="maximum training epochs")
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help="batch size")
parser.add_argument('--dropout', type=float, default=0.3,
                    help="dropout applied to layers (0 = no dropout)")
parser.add_argument('--alpha', type=float, default=1.0,
                    help="weight of the mutual information term")
parser.add_argument('--beta', type=float, default=3.0,
                    help="topic latent weight")
parser.add_argument('--gamma', type=float, default=0.3,
                    help="weight of the discriminator")
parser.add_argument('--sigma', type=float, default=0,#5e5,
                    help="weight of mmd(latent, latent_prior)")
parser.add_argument('--srec_w', type=float, default=0.0,
                    help="weight of srec loss of z_t with Gaussian distribution")
parser.add_argument('--rec_w', type=float, default=0.0,
                    help="weight of rec loss of z_t with Gaussian distribution")
parser.add_argument('--lr', type=float, default=1e-4,
                    help="learning rate")
parser.add_argument('--wd', type=float, default=1e-5,
                    help="weight decay used for regularization")
parser.add_argument('--tc_weight', type=float, default=0.0,
                    help="total correlation weight")
parser.add_argument('--clip', type=float, default=5.0,
                    help="max clip norm of gradient clip")
parser.add_argument('--flow_type', type=str, default='hhf',
                    help="Choose hhf or nmt flow")
parser.add_argument('--t_type', type=str, default='dirichlet',
                    help="Choose normal or dirichlet to model topic latent variable")
parser.add_argument('--mmd_type', type=str, default='t',
                    help="Choose t or z for mmd mutual information calculation")
parser.add_argument('--flow_num_layer', type=int, default=10,
                    help="Householder flow layer number")
parser.add_argument('--epoch_size', type=int, default=2000,
                    help="number of training steps in an epoch")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed")
parser.add_argument('--kla', type=str, default='cyc',
                    help="use kl annealing cyc or mono")
parser.add_argument('--ckpt', type=str, default=None,
                    help="name of loaded check point")
parser.add_argument('--nocuda', action='store_true',
                    help="do not use CUDA")
args = parser.parse_known_args()[0]

torch.manual_seed(args.seed)
random.seed(args.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
now = datetime.datetime.now()

def evaluate(data_iter, model, pad_id, ignore_index, mmd_type):
    model.eval()
    data_iter.init_epoch()
    size = len(data_iter.data())
    seq_loss = 0.0
    bow_loss = 0.0
    discri_loss = 0.0
    tc_loss = 0.0
    kld_z = 0.0
    kld_t = 0.0
    weighted_mmd_loss = 0.0
    seq_words = 0
    bow_words = 0
    alpha_entrophy = []
    mmd_weight = args.sigma
    for batch in data_iter:
        texts, lengths = batch.text
        batch_size = texts.size(0)
        inputs = texts[:, :-1].clone()
        targets = texts[:, 1:].clone()
        results = model(inputs, lengths - 1, ignore_index)
        batch_seq = seq_recon_loss(
            results.seq_outputs, targets, pad_id
        )
        batch_bow = bow_recon_loss(
            results.bow_outputs, results.bow_targets
        )
        # Discriminator Loss --> NTM augmented
        batch_discri = discriminator_loss(
            results.discri_log, results.bow_targets
        )
        if args.flow_type == 'hhf':
            batch_kld_z = hhf_kld(results.z_start, results.z)
            # batch_kld_z = 3200 * hhf_mmd_kld(results.z_start, results.z)
        elif args.flow_type == 'nmt':
            batch_kld_z = nmt_kld(results.nmt_logp, results.z)
        else:
            batch_kld_z = normal_kld(results.posterior_z)
        batch_kld_t = normal_kld(results.posterior_t,
                                results.prior_t).to(inputs.device)

        batch_tc_loss = cal_tc(results.D_zt, results.D_zt_permute)

        if mmd_type == 'z':
            latent = results.posterior_z.mean
            latent_prior = torch.randn_like(latent)
        elif mmd_type == 't':
            latent = results.posterior_t.mean.to(inputs.device)
            latent_prior = results.prior_t.mean.to(inputs.device)
        else:
            z = results.posterior_z.mean
            t = results.posterior_t.mean.to(z.device)
            latent = torch.cat([z, t], dim=1)
            t_prior = results.prior_t.mean.to(inputs.device)
            z_prior = torch.randn_like(z)
            latent_prior = torch.cat([z_prior, t_prior], dim=1)
        batch_mmd_loss = mmd_v2(latent, latent_prior)

        t = results.posterior_t.rsample() # VRTM used sample method to calculate entrophy (.mean is another option)
        alpha_entrophy.append(float(torch.mean(torch.sum(-t * torch.log(t + 1e-10), -1)).detach().cpu()))

        seq_loss += batch_seq.item() / size
        bow_loss += batch_bow.item() / size
        discri_loss += batch_discri.item() / size
        kld_z += batch_kld_z.item() / size
        kld_t += batch_kld_t.item() / size
        tc_loss += batch_tc_loss.item() / size
        weighted_mmd_loss += mmd_weight * batch_mmd_loss.item()
        seq_words += torch.sum(lengths - 1).item()
        bow_words += torch.sum(results.bow_targets)
    seq_ppl = math.exp(seq_loss * size / seq_words)
    bow_ppl = math.exp(bow_loss * size / bow_words)
    return (seq_loss, bow_loss, kld_z, kld_t,
            seq_ppl, bow_ppl, discri_loss, tc_loss, weighted_mmd_loss, alpha_entrophy)


def train(data_iter, neg_data_iter, model, pad_id, ignore_index, optimizer, epoch, mmd_type, neg_sample, t_type):
    model.train()
    data_iter.init_epoch()
    neg_data_iter.init_epoch()
    size = min(len(data_iter.data()), args.epoch_size * args.batch_size)
    seq_loss = 0.0
    bow_loss = 0.0
    discri_loss = 0.0
    tc_loss = 0.0
    kld_z = 0.0
    kld_t = 0.0
    weighted_mmd_loss = 0.0
    seq_words = 0
    bow_words = 0
    srec_loss = 0
    rec_loss = 0
    alpha_entrophy = []
    mmd_weight = args.sigma
    i = 0
    for batch in data_iter:
        if i == args.epoch_size:
            break

        if t_type == 'normal':
            neg_texts = get_neg_batch(data_iter, neg_sample, pad_id)
            if batch.text[0].size(0) != neg_texts.size(0) * neg_sample:
                continue
            neg_inputs = neg_texts[:, :-1].clone()

        # neg_texts: batch tensor, size [neg_sampls * batch_size, padded length]
        # batch: BucketIterator object
        texts, lengths = batch.text
        batch_size = texts.size(0)
        inputs = texts[:, :-1].clone()
        targets = texts[:, 1:].clone()
        if t_type == 'normal':
            print(f"start {i}")
            results = model.inference(inputs, neg_inputs, neg_sample, lengths - 1, ignore_index)
            print(f"finish {i}")
        else:
            results = model(inputs, lengths - 1, ignore_index)
        ##########################################################
        #########-----------Calculate loss--------------##########
        ##########################################################
        # sequence reconstruction loss <-- \max\log(p(x|z, t)) + \max\log(p(x|t))
        batch_seq = seq_recon_loss(
            results.seq_outputs, targets, pad_id
        )  # + seq_recon_loss(results.t_seq_outputs, targets, pad_id)
        # Bag of Word loss <-- prevent KL collapse
        batch_bow = bow_recon_loss(
            results.bow_outputs, results.bow_targets
        )
        # Discriminator Loss --> NTM augmented
        batch_discri = discriminator_loss(
            results.discri_log, results.bow_targets
        )
        # batch_discri = torch.tensor([0], device=inputs.device)

        # KLD loss of 2 latents
        if args.flow_type == 'hhf':
            batch_kld_z = hhf_kld(results.z_start, results.z)
            # batch_kld_z = 3200 * hhf_mmd_kld(results.z_start, results.z)
        elif args.flow_type == 'nmt':
            batch_kld_z = nmt_kld(results.nmt_logp, results.z)
        else:
            batch_kld_z = normal_kld(results.posterior_z)
        batch_kld_t = normal_kld(results.posterior_t,
                                results.prior_t).to(inputs.device)

        batch_tc_loss = cal_tc(results.D_zt, results.D_zt_permute)

        if mmd_type == 'z':
            latent = results.posterior_z.rsample()
            latent_prior = torch.randn_like(latent)
        elif mmd_type == 't':
            latent = results.posterior_t.rsample().to(inputs.device)
            latent_prior = results.prior_t.rsample().to(inputs.device)
        else:
            z = results.posterior_z.rsample()
            t = results.posterior_t.rsample().to(z.device)
            latent = torch.cat([z, t], dim=1)
            t_prior = results.prior_t.rsample().to(inputs.device)
            z_prior = torch.randn_like(z)
            latent_prior = torch.cat([z_prior, t_prior], dim=1)
        batch_mmd_loss = mmd_v2(latent, latent_prior)

        t = results.posterior_t.rsample()
        alpha_entrophy.append(float(torch.mean(torch.sum(-t * torch.log(t + 1e-10), -1)).detach().cpu()))

        if t_type == 'normal':
            # batch_reg_loss = 0.0
            batch_reg_loss = results.reg_loss
            # batch_srec_loss = 0.0
            raw_loss = torch.clamp(1 - results.pos_scores + results.neg_scores, min=0.).mean(0)
            batch_srec_loss = raw_loss.mean()
        else:
            batch_srec_loss = 0.0
            batch_reg_loss = 0.0

        seq_loss += batch_seq.item() / size
        bow_loss += batch_bow.item() / size
        discri_loss += batch_discri.item() / size
        kld_z += batch_kld_z.item() / size
        kld_t += batch_kld_t.item() / size
        tc_loss += batch_tc_loss.item() / size  # no batch_size divided
        weighted_mmd_loss += mmd_weight * batch_mmd_loss.item()
        srec_loss += batch_srec_loss / size
        rec_loss += batch_reg_loss / size
        seq_words += torch.sum(lengths - 1).item()
        bow_words += torch.sum(results.bow_targets)
        if args.kla == 'mono':
            kld_weight = weight_schedule_mono(args.epoch_size * (epoch - 1) + i)
        elif args.kla == 'cyc':
            # kld_weight = weight_schedule_cyclical_duration(args.epoch_size * (epoch - 1) + i)
            kld_weight = weight_schedule_cyclical_epoch(epoch)
        else:
            kld_weight = 1.
        # todo: anneal discriminator loss weight
        discri_weight = args.gamma #weight_schedule(args.epoch_size * (epoch - 1) + i) if args.kla else 1.
        tc_weight = args.tc_weight  # weight_schedule(args.epoch_size * (epoch - 1) + i) if args.kla else 1.
        srec_w = args.srec_w
        rec_w = args.rec_w
        optimizer.zero_grad()
        kld_term = (batch_kld_z + batch_kld_t * args.beta) / batch_size
        loss_fet = (batch_seq + batch_bow * args.beta + batch_discri * discri_weight +
                tc_weight * batch_tc_loss + mmd_weight * batch_mmd_loss + rec_w * rec_loss + srec_w * srec_loss
                    ) / batch_size + kld_weight * kld_term
        loss_fet.backward()
        #
        # loss = (rec_w * rec_loss) / batch_size + loss_fet
        # loss.backward(retain_graph=True)
        clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        i += 1
    seq_ppl = math.exp(seq_loss * size / seq_words)
    bow_ppl = math.exp(bow_loss * size / bow_words)

    return (seq_loss, bow_loss, kld_z, kld_t,
            seq_ppl, bow_ppl, discri_loss, tc_loss, kld_weight, weighted_mmd_loss, srec_loss, rec_loss, alpha_entrophy)


def interpolate(i, start, duration):
    return max(min((i - start) / duration, 1), 0)


def weight_schedule_mono(t):
    """Scheduling of the KLD annealing weight monotonicly. """
    return interpolate(t, 6000, 40000)

def weight_schedule_cyclical_duration(t):
    """Scheduling of the KLD annealing weight cyclically. """
    total_iter = args.epoch_size * args.epochs
    partition = total_iter / 4 # cyclical time
    duration = (3/4) * partition - 6000 if t <= partition else (2/4) * partition
    start = 6000 if t <= partition else 0
    return interpolate(t, start, duration)
def weight_schedule_cyclical_epoch(cur_epoch):
    total_epoch = args.epochs
    partition = int(total_epoch / 4)
    overflow = cur_epoch % partition
    growstage = int(partition * (2/5))
    maintainstage = partition - growstage
    if cur_epoch <=3:
        weight = 0.
    elif overflow < growstage:
        weight = (growstage - overflow)/growstage
    else:
        weight = 1.
    return weight


# model saver
def get_savepath(args, epoch):
    dataset = args.data
    ckpt_root = f'./ckpt/{dataset}'
    os.makedirs(ckpt_root, exist_ok=True)
    path = ckpt_root + '/emb{0:d}.hid{1:d}.z{2:d}.t{3:d}_{4}.{5}.kla_{6}.' \
                       'tc{7:.1f}.beta{8:.1f}.{9}.flow.{10}.{11}epoch.date{12}.' \
                       '{13}.mmdw{14}-{15}.disw{16}.minfreq{17}.flownum{18}.pt'.format(args.embed_size, args.hidden_size,
                                                 args.code_size, args.num_topics, args.t_type,'.wd{:.0e}'.format(args.wd) if args.wd > 0 else '',
                                                                         args.kla, args.alpha, args.beta, args.flow_type, dataset, epoch,
                                                                         now.month, now.day, args.sigma, args.mmd_type, args.gamma, args.min_freq, args.flow_num_layer)
    return path

def get_last_pt(args):
    dataset = args.data
    ckpt_root = f'./ckpt/{dataset}'
    pt_list = os.listdir(ckpt_root)
    pt_list.sort(key=lambda x: int(x.split('.')[-2].split('epoch')[0]))
    return ckpt_root + str(pt_list[-1])


def main(args):
    resume_file = args.ckpt
    neg_sample = 1
    dataset = args.data
    min_freq = args.min_freq
    # logger
    log_file = f"./log/{dataset}_z{args.hidden_size}_t{args.hidden_size_t}_{args.t_type}_beta{args.beta}_mmdw{args.sigma}-{args.mmd_type}_disw{args.gamma}" \
               f"_tcw{args.alpha}_zsize{args.code_size}_tsize{args.num_topics}_flow-{args.flow_type}.{args.flow_num_layer}" \
               f"_kla{args.kla}_minfreq{min_freq}_{now.month}.{now.day}.txt"
    # log_file = f"./log/test.txt"
    logger = Logger(log_file)

    logger.info("Loading data")
    if dataset in ['yahoo', 'yelp', 'yahoo_ques', 'yahoo_ans']:
        with_label = True
        max_vocab = 20000
        max_length = 150
    elif dataset in ['imdb_s']:
        with_label = True
        max_vocab = 40000
        max_length = 80
    else:
        with_label = False
        max_vocab = 40000
        max_length = 80
    corpus = Corpus(
        f'./data/{args.data}', max_vocab_size=max_vocab, min_freq=min_freq,
        max_length=max_length, with_label=with_label
    )
    pad_id = corpus.word2idx[PAD_TOKEN]
    sos_id = corpus.word2idx[SOS_TOKEN]
    vocab_size = len(corpus.word2idx)
    with open(args.splist, 'r') as f:
        xx = f.read()
    stop_list = [corpus.word2idx[spword] for spword in xx.split('\n')]
    stop_list.append(pad_id)
    logger.info(f"\ttraining data size: {len(corpus.train)}")
    logger.info(f"\tvocabulary size: {vocab_size}")
    logger.info("Constructing model")
    logger.info(args)
    device = 'cuda'  # torch.device('cpu' if args.nocuda else 'cuda')
    flow = False if args.flow_type is None else True
    model = TopGenVAE(
        vocab_size, args.embed_size, args.hidden_size, args.hidden_size_t, args.t_type, args.code_size,
        args.num_topics, args.dropout, args.flow_num_layer, flow, device
    )
#     model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    start_epoch = 1
    if resume_file is not None:
        checkpoint = torch.load(f'./ckpt/{args.data}/{resume_file}')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    best_loss = None
    print('model structure:')
    print(model)
    train_iter = get_iterator(corpus.train, args.batch_size, True, device)
    # train_neg_iter = get_iterator(corpus.train, args.batch_size * neg_sample, True, device)
    valid_iter = get_iterator(corpus.valid, args.batch_size, False, device)
    test_iter = get_iterator(corpus.test, args.batch_size, False, device)
    logger.info("\nStart training")
    ##########################################################
    #########-----------Start Training--------------##########
    ##########################################################
    try:
        for epoch in range(start_epoch, start_epoch + args.epochs + 1):
            # todo: anneal discriminator loss weight
            discri_weight = args.gamma #weight_schedule(args.epoch_size * (epoch - 1) + i) if args.kla else 1.
            tc_weight = args.tc_weight  # weight_schedule(args.epoch_size * (epoch - 1) + i) if args.kla else 1.
            epoch_start_time = time.time()
            (tr_seq_loss, tr_bow_loss, tr_kld_z, tr_kld_t,
             tr_seq_ppl, tr_bow_ppl, tr_discri_loss, tr_tc_loss, kld_weight, tr_weighted_mmd, tr_srec_loss, tr_rec_loss, tr_entrophy) \
                = train(train_iter, train_iter, model, pad_id, stop_list, optimizer, epoch, args.mmd_type, neg_sample, args.t_type)
            (va_seq_loss, va_bow_loss, va_kld_z, va_kld_t,
             va_seq_ppl, va_bow_ppl, va_discri_loss, va_tc_loss, va_weighted_mmd, va_entrophy) = evaluate(
                valid_iter, model, pad_id, stop_list, args.mmd_type
            )
            logger.info('-' * 90)
            meta = "| epoch {:2d} | time {:5.2f}s ".format(epoch, time.time() - epoch_start_time)
            logger.info(
                meta + "| train loss {:5.2f} {:5.2f} ({:5.2f} {:5.2f}) | Dis / TC / MMD loss {:5.2f} / {:5.2f} / {:5.2f} | srec / reg loss {:5.2f} / {:5.2f}"
                       " | train ppl {:5.2f} {:5.2f} | Ent {:5.2f} + {:5.2f} / {:5.2f}".format(
                    tr_seq_loss, tr_bow_loss, tr_kld_z, tr_kld_t, tr_discri_loss, tr_tc_loss, tr_weighted_mmd, tr_srec_loss, tr_rec_loss,
                    tr_seq_ppl, tr_bow_ppl, np.mean(tr_entrophy), np.std(tr_entrophy), np.min(tr_entrophy)))
            logger.info(len(
                meta) * ' ' + "| valid loss {:5.2f} {:5.2f} ({:5.2f} {:5.2f}) | Dis / TC / MMD loss {:5.2f} / {:5.2f} / {:5.2f}"
                              " | valid ppl {:5.2f} {:5.2f} | Ent {:5.2f} + {:5.2f} / {:5.2f}".format(
                va_seq_loss, va_bow_loss, va_kld_z, va_kld_t, va_discri_loss, va_tc_loss, va_weighted_mmd,
                va_seq_ppl, va_bow_ppl, np.mean(va_entrophy), np.std(va_entrophy), np.min(va_entrophy)))
            # epoch_loss = va_seq_loss + va_bow_loss + kld_weight * (va_kld_z + args.beta * va_kld_t) + \
            #              discri_weight * va_discri_loss + tc_weight * va_tc_loss + va_weighted_mmd
            epoch_loss = va_seq_loss + va_bow_loss + (va_kld_z + args.beta * va_kld_t) + discri_weight * va_discri_loss\
                         + tc_weight * va_tc_loss + va_weighted_mmd
            logger.info("epoch loss {:5.2f}".format(epoch_loss))
            if best_loss is None or epoch_loss < best_loss or epoch >= start_epoch + args.epochs - 5:
                if epoch_loss >=0 and va_kld_z>=0:
                    best_loss = epoch_loss
                    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                    with open(get_savepath(args, epoch), 'wb') as f:
                        torch.save(state, f)
                    with open(get_savepath(args, epoch), 'rb') as f:
                        ckpt = torch.load(f)
                        model.load_state_dict(ckpt['model'])
                    (te_seq_loss, te_bow_loss, te_kld_z, te_kld_t,
                     te_seq_ppl, te_bow_ppl, te_discri_loss, te_tc_loss, te_weighted_mmd_loss, te_entrophy) = evaluate(test_iter,
                                                                                                          model, pad_id,
                                                                                                          stop_list,
                                                                                                          args.mmd_type)
                    logger.info('=' * 90)
                    logger.info(
                        " | test loss {:5.2f} {:5.2f} ({:5.2f} {:5.2f}) | Dis / TC / MMD loss {:5.2f} / {:5.2f} / {:5.2f}"
                        " | test ppl {:5.2f} {:5.2f} | Ent {:5.2f} + {:5.2f} / {:5.2f}".format(
                            te_seq_loss, te_bow_loss, te_kld_z, te_kld_t, te_discri_loss, te_tc_loss, te_weighted_mmd_loss,
                            te_seq_ppl, te_bow_ppl, np.mean(te_entrophy), np.std(te_entrophy), np.min(te_entrophy)))
                    logger.info('=' * 90)
                word_id = model.sample(2, 20, sos_id, device)
                word_sent = [[corpus.idx2word[w] for w in sen] for sen in word_id]
                for sent in word_sent:
                    logger.info(" ".join(str(i) for i in sent))
            # save model at last epoch
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        with open(get_savepath(args, epoch), 'wb') as f:
            torch.save(state, f)
        word_id = model.sample(2, 100, sos_id, device)
        word_sent = [[corpus.idx2word[w] for w in sen] for sen in word_id]
        for sent in word_sent:
            logger.info(" ".join(str(i) for i in sent))

    except KeyboardInterrupt:
        logger.info('-' * 90)
        logger.info('Exiting from training early')

    # with open(get_savepath(args,epoch), 'rb') as f:
    #     ckpt = torch.load(f)
    #     model.load_state_dict(ckpt['model'])
    # (te_seq_loss, te_bow_loss, te_kld_z, te_kld_t,
    #  te_seq_ppl, te_bow_ppl, te_discri_loss, te_tc_loss, te_weighted_mmd_loss) = evaluate(test_iter, model, pad_id, stop_list, args.mmd_type)
    # logger.info('=' * 90)
    # logger.info(
    #     "| End of training | test loss {:5.2f} {:5.2f} ({:5.2f} {:5.2f}) | Discriminator / TC / MMD loss {:5.2f} / {:5.2f} / {:5.2f}"
    #     " | test ppl {:5.2f} {:5.2f}".format(
    #         te_seq_loss, te_bow_loss, te_kld_z, te_kld_t, te_discri_loss, te_tc_loss, te_weighted_mmd_loss,
    #         te_seq_ppl, te_bow_ppl))
    # logger.info('=' * 90)


if __name__ == '__main__':
    main(args)
