# runing example for en-vi
set -x

exp_name=bert-base-uncased_to_gerbert

ARTIFACT_DIR=/fs/cml-projects/vocab-swap/shared/artifacts/align_$exp_name

VEC_PATH=$ARTIFACT_DIR/vecs
VOCAB_PATH=$ARTIFACT_DIR/tokenizers
PROB_PATH=$ARTIFACT_DIR/probs

mkdir -p $ARTIFACT_DIR
mkdir -p $VEC_PATH
mkdir -p $VOCAB_PATH
mkdir -p $PROB_PATH

# For alignment code
src_lg=en
tgt_lg=de

langs=($src_lg $tgt_lg)
max_load=100000

# for huggingface tokenizer retrieval
src_name=bert-base-uncased
tgt_name=gerbert
src_hf_name=$src_name
tgt_hf_name=dbmdz/bert-base-german-cased
BASE_TOKENIZER_SAVE_PATH=$VOCAB_PATH

for lg in ${langs[@]}; do
  wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.${lg}.300.vec.gz -P $VEC_PATH
  gunzip $VEC_PATH/cc.${lg}.300.vec.gz
  # get wordlist
  head -$max_load $VEC_PATH/cc.$lg.300.vec > $VEC_PATH/cc.$lg.300.vec.$max_load
  cut -f1 -d ' ' $VEC_PATH/cc.$lg.300.vec.$max_load > $VEC_PATH/wordlist.$lg
done

# # # note that it's important to  have wordlist.en, where en is the typical source language, as the first argument
python get_common_words.py $VEC_PATH/wordlist.$src_lg $VEC_PATH/wordlist.$tgt_lg > $VEC_PATH/wordlist.$src_lg-$tgt_lg

# fasttext alignment script abs path
ALIGN=/cmlscratch/jkirchen/vocab-root/fastText/alignment/align.py

python $ALIGN --src_emb $VEC_PATH/cc.$tgt_lg.300.vec --tgt_emb $VEC_PATH/cc.$src_lg.300.vec --dico_train $VEC_PATH/wordlist.$src_lg-${lg} --dico_test $VEC_PATH/wordlist.$src_lg-$tgt_lg --output $VEC_PATH/aligned.$tgt_lg.vec  --niter 10 --maxload $max_load

python get_prob_vect.py --src_aligned_vec $VEC_PATH/cc.$src_lg.300.vec --src_tokenizer $src_hf_name --topn 50000 --tgt_aligned_vec $VEC_PATH/aligned.$tgt_lg.vec --tgt_tokenizer $tgt_hf_name --save $PROB_PATH/probs.mono.$src_name-to-$tgt_name.pth
