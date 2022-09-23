#! /usr/bin/env bash

set -x;

# ... removed all steps to produce the translation matrix

lg=de

DATA=../../data
UTILS=../utils
ARTIFACTS=$DATA/artifacts

SRC_VOCAB_PATH=$ARTIFACTS/en/tokenizer
TGT_VOCAB_PATH=$ARTIFACTS/$lg/tokenizer

SRC_PRETRAINED_PATH=$ARTIFACTS/en/model
TGT_PRETRAINED_PATH=$ARTIFACTS/$lg/model

PROB_PATH=$ARTIFACTS/probs


echo "initialize target model"
python $UTILS/init_weight.py --src_vocab $SRC_VOCAB_PATH/en-vocab.txt --src_model $SRC_PRETRAINED_PATH/pytorch_model.bin --prob $PROB_PATH/probs.mono.en-$lg.pth --tgt_model $TGT_PRETRAINED_PATH/pytorch_model.bin --tgt_vocab $TGT_VOCAB_PATH/$lg-vocab.txt

# NOTE: you have to change the vovab_size manually in $TGT_INIT/config.json file 
# NOTE-NOTE: it should be correct but probably want to verify the vocabulary sizes? 
