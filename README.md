# FET-LM: Flow Enhanced Variational Auto-Encoder for Topic-Guided Language Modeling

Official PyTorch Implementation of  *[FET-LM]()*, accepted to *[IEEE Transactions on Neural Networks and Learning Systems (TNNLS)](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385)*. We provide (1) the code of FET-LM over 5 datasets (APNEWS, BNC, IMDB, PTB and Yelp15) on language modeling and (2) the full paper [here](https://github.com/ImKeTT/FET-LM/paper/fet-lm.pdf) (including full Appendix).

## Setup

### Environment

Make sure you have installed the following packages:

```bash
torch
torchtext
nltk
```

### Word Vector

Download glove vector from http://nlp.stanford.edu/data/glove.640B.300d.zip. Put it under the path `/home/user/.vector_cache`, or wherever you like but don't forget to change the data path accordingly in `data_loader.py`.

### Data Preparation

1. Download APNEWS, IMDB, BNC dataset from the existing work:  https://github.com/jhlau/topically-driven-language-model. 
2. Download PTB dataset from https://deepai.org/dataset/penn-treebank. 
3. Doenload Yelp15 review dataset from https://www.yelp.com/dataset.  

Rename text files and make sure there exist `train.txt` for training,  `valid.txt` for validation and `test.txt` for testing in each data folder. Put respective folder in `./data` folder. 

## Training

For training, PPL and topic entropy results will show in this process. 

```bash
# e.g., train on APNEWS
python train.py --data apnews --min_freq 2 --epochs 80 --sigma 5e2 --gamma 0.3 --flow_num_layer 10 --kla cyc --num_topics 50
```

Detailed training description of each dataset please refer to `./config/{dataset name}.json` , add the commands to bash line accordingly. Will incorporate these config files into training code soon~

```json
// e.g., for dataset APNEWS
{
    "data":"apnews",
    "splist": "./data/stopwords.txt", // location of the stopwords file
    "max_vocab":40000, // maximum vocabulary size for the input
    "max_length":80, // maximum sequence size for the input
    "embed_size":200, // size of the word embedding
    "hidden_size":300, // number of hidden units for z_s
    "hidden_size_t":200, // number of hidden units for z_t
    "code_size":32, // latent code dimension
    "num_topics":50, // topic number, choose from 10 / 30 / 50
    "min_freq":2, // min frequency for corpus
    "epochs":80, // training epoch
    "batch_size":32, // batch size
    "dropout":0.3, // dropout rate for all RNN
    "alpha":1.0, // weight of the mutual information term
    "beta":3.0, // weight of the topic component
    "gamma":0.3, // weight of the discriminator
    "sigma":5e5, // weight of the info penalty mmd(latent, latent_prior)
    "srec_w":0.0, // weight of srec loss of z_t with Gaussian distribution
    "rec_w":0.0, // weight of rec loss of z_t with Gaussian distribution
    "lr":1e-4, // learning rate
    "wd":1e-5, // weight decay used for regularization
    "tc_weight":0.0, // total correlation weight
    "clip":5.0, // max clip norm of gradient clip
    "flow_type":"hhf", // flow type for sequential posterior modeling
    "t_type":"dirichlet", // type of z_t, choose from "dirichlet" / "normal"
    "mmd_type":"t", // type on which info penalty is put, choose from "t" / "z" / "all"
    "flow_num_layer":10, // number of flow layer
    "epoch_size":2000, // number of training steps in an epoch
    "seed":42, //random seed
    "kla":"cyc", // use kl annealing "cyc" or "mono"
    "ckpt":""// name of loaded check point

}
```

## Testing

TBD.

## Others

Please email me or open an issue if you have any question.

if you find our work useful, please star the repo and cite the paper :)

```
TBD
```

We thank open sourced codes related to VAEs and flow-based methods, which inspired our work !!
