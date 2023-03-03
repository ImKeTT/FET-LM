import torch
from torchtext.data import Field, LabelField
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator
from torchtext.vocab import Vectors

PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'


class Corpus(object):
    def __init__(self, datadir, min_freq=2, max_vocab_size=None, max_length=None, with_label=False, stop_words=None, with_ques=False):
        """
        word corpus construction
        :param datadir:
        :param min_freq: least frequency needed to be include into vocab
        :param max_vocab_size:
        :param max_length:
        :param with_label:
        :param stop_words: stop words list
        """
        tokenize = lambda x: x.split() # default tokenizer
        if max_length is None:
            preprocessing = None
        else:
            preprocessing = lambda x: [ii.replace("\t", " ") for ii in x][:max_length]
        TEXT = Field(
            sequential=True, tokenize='spacy',
            init_token=SOS_TOKEN, eos_token=EOS_TOKEN,
            pad_token=PAD_TOKEN, unk_token=UNK_TOKEN,
            preprocessing=preprocessing, lower=True,
            include_lengths=True, batch_first=True,
            stop_words=stop_words
        )
        LABEL = Field(sequential=False, use_vocab=False)
        # LABEL = LabelField(sequential=False, dtype=torch.float)
        if with_label:
            datafields = [('label', LABEL), ('text', TEXT)]
        else:
            datafields = [('text', TEXT)]
        self.train, self.valid, self.test = TabularDataset.splits(
            path=datadir, train='train.txt', validation='valid.txt',
            test='test.txt', format='tsv', fields=datafields
        )
        cache = '/home/user/.vector_cache'
        vectors = Vectors(name='glove.6B.300d.txt', cache=cache)
        TEXT.build_vocab(
            self.train, self.valid, max_size=max_vocab_size,
            min_freq=min_freq, vectors=vectors
        )
        self.word2idx = TEXT.vocab.stoi
        self.idx2word = TEXT.vocab.itos
        self.with_label = with_label


def get_iterator(dataset, batch_size, train, device):
    """
    iterator object for data utilization
    :param dataset: dataset name
    :param batch_size: batch size
    :param train: shuffle is true if train
    :param device: cuda or cpu
    :return:
    """
    sort_key = lambda x: len(x.text)
    dataset_iter = BucketIterator(
        dataset, batch_size=batch_size, device=device,
        train=train, shuffle=train, repeat=False,
        sort_key=sort_key, sort_within_batch=True
    )
    return dataset_iter

def get_neg_batch(data_iter, neg_samples, pad_id):
    max_len = 0
    neg_batch = []
    for i, batch in enumerate(data_iter):
        if i == neg_samples:
            break
        inputs, lengths = batch.text
        if lengths[0].item() > max_len:
            max_len = lengths[0].item()
        neg_batch.append(inputs)
    for ii in range(len(neg_batch)):
        dist = max_len - neg_batch[ii].size(1)
        if dist > 0:
            padding = torch.full((neg_batch[ii].size(0), dist), pad_id, device=neg_batch[0].device).long()
            neg_batch[ii] = torch.cat([neg_batch[ii], padding], dim=-1)
    return torch.cat(neg_batch, dim=0)

if __name__=="__main__":
    tokenizer = lambda x: x.split()
    TEXT = Field(tokenize='spacy', lower=True)
    LABEL = LabelField(sequential=False, dtype=torch.float)

    trainds = TabularDataset(
        path='./data/yelp/test.txt', format='tsv',
        fields=[('label', LABEL), ('text', TEXT)], skip_header=True)
    TEXT.build_vocab(trainds)