# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import argparse
import torch
from collections import defaultdict
from transformers import AutoTokenizer
import numpy as np
from sparsemax import Sparsemax
import io
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Estimate word translation probability from aligned fastText vectors")

parser.add_argument('--src_aligned_vec', default='wiki.en.align.vec',
                    help="source embeddings in aligned space.")
parser.add_argument('--tgt_aligned_vec', default='',
                    help="source embeddings in aligned space.")
parser.add_argument('--src_tokenizer', default='',
                    help='name or path of src hf tokenizer')
parser.add_argument('--tgt_tokenizer', default='',
                    help='name or path of tgt hf tokenizer')
# parser.add_argument('--src_vocab', default='',
#                     help='path to src vocabulary')
# parser.add_argument('--src_merge', default='',
#                     help='path to src vocabulary merge file')
# parser.add_argument('--tgt_vocab', default='',
#                     help="target vocabulary file.")
# parser.add_argument('--tgt_merge', default='',
#                     help='path to tgt vocabulary merge file')
parser.add_argument('--topn', type=int, default=100000,
                    help="max number of words in the vocab")
parser.add_argument('--save', default='', help='path to save file')


def load_vec(fname, nmax=50000):
    vectors = []
    word2id = {}
    print(f'| load top {nmax} word vectors from {fname}')
    with io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = torch.from_numpy(np.stack(vectors)).float()
    return embeddings, id2word, word2id


def get_subword_embeddings(tokenizer, fname, nmax):
    embeddings, id2word, word2id = load_vec(fname, nmax)
    rank =  torch.arange(1, nmax+1).float()
    invf = 1.0 / rank  # invert rank to approximate power-law distribution

    token_to_id = tokenizer.get_vocab()
    unk_token_id = tokenizer(f"{tokenizer.unk_token}")["input_ids"][1]

    sub2wids = defaultdict(list)
    ntot = len(word2id)
    nuns = 0
    for w, i in word2id.items():
        tokens = tokenizer.tokenize(f' {w}')  # add space to handle Roberta tokenizer # should be good here?
        token_ids = tokenizer(f' {w}')["input_ids"][1:-1]
        if len(tokens) == 1: nuns += 1
        for t,t_id in zip(tokens,token_ids):
            if t != tokenizer.unk_token and t_id != unk_token_id:
                sub2wids[t].append(i)
                try:
                    token_to_id[t]
                except:
                    breakpoint()

    # report
    print(f'| {ntot} words - {nuns} unsegmented words')
    print(f'| Using {len(sub2wids.items())} unique subwords to represent all words in the vec file (fasttext) word list')
    print(f"| out of {tokenizer.vocab_size} total in the real tokenizer vocabulary")        


    # compute sub-word embeddings
    sub_embs = []
    subwords = []
    for k, wids  in sub2wids.items():
        vec = embeddings[wids]
        uni_prob = invf[wids] / invf[wids].sum()   # normalize to get probability
        sub_vec = vec.t() @ uni_prob
        sub_embs.append(sub_vec)
        subwords.append(k)
    sub_embs = torch.stack(sub_embs)
    return sub_embs, subwords

def renorm(x, dim):
    # re-normalize x
    y = torch.norm(x, dim=dim, keepdim=True)
    return x/y

def main():
    params = parser.parse_args()
    print(params)

    try:
        # requires a valid hf name
        src_tokenizer = AutoTokenizer.from_pretrained(params.src_tokenizer)
    except Exception as e:
        raise e(f"Make sure to pass a valid name or path for a hf tokenizer.")
    try:
        tgt_tokenizer = AutoTokenizer.from_pretrained(params.tgt_tokenizer)
    except Exception as e:
        raise e(f"Make sure to pass a valid name or path for a hf tokenizer.")

    src_embs, src_subwords = get_subword_embeddings(src_tokenizer, params.src_aligned_vec, params.topn)
    tgt_embs, tgt_subwords = get_subword_embeddings(tgt_tokenizer, params.tgt_aligned_vec, params.topn)
    src_embs = renorm(src_embs, 1)
    tgt_embs = renorm(tgt_embs, 1)
    # initialize sparse-max

    sparsemax = Sparsemax(1)

    print(f'| # src subwords founds: {len(src_subwords)}')
    print(f'| # tgt subwords founds: {len(tgt_subwords)}')
    print('| compute translation probability')
    scores = tgt_embs @ src_embs.t()
    a = sparsemax(scores)    # (Vf, Ve)
    print('| generating translation table!')
    probs = {}

    for i, tt in tqdm(enumerate(tgt_subwords), total=len(tgt_subwords)):
        probs[tt] = {}
        ix = torch.nonzero(a[i]).view(-1)
        px = a[i][ix].tolist()
        wx = [src_subwords[j] for j in ix.tolist()]
        probs[tt] = {w: p for w, p in zip(wx, px)}
    n_avg = np.mean([len(ss) for ss in probs.values()])
    print(f'| average # source / target: {n_avg:.2f} ')
    print(f"| save translation probabilities: {params.save}")
    torch.save(probs, params.save)


if __name__ == '__main__':
    main()
