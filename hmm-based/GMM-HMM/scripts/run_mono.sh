#!/bin/bash

echo -e ""
echo -e "Let's start..."
echo -e ""

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# Enlaces simbolicos a los scripts de Kaldi
#ln -s ../../wsj/s5/utils/
#ln -s ../../wsj/s5/steps/

nj=10 # numero de jobs en paralelo

data_dir=$1
train=$2
num_iters=$3
totgauss=$4
lang_dir=$5
output_dir=$6/numiters${num_iters}/totgauss${totgauss}/

echo -e ""
echo -e "==== TRAINING MONOPHONE SYSTEM ===="
echo -e ""

steps/train_mono.sh --nj $nj --cmd "$train_cmd" --boost-silence 1.25 --num-iters $num_iters --totgauss $totgauss ${data_dir}/${train} ${lang_dir} \
  ${output_dir}/mono || exit 1;

utils/mkgraph.sh ${lang_dir} ${output_dir}/mono ${output_dir}/mono/graph
#local/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" ${output_dir}/mono/graph ${data_dir}/train ${output_dir}/mono/decode_train
local/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" ${output_dir}/mono/graph ${data_dir}/dev ${output_dir}/mono/decode_dev
local/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" ${output_dir}/mono/graph ${data_dir}/test ${output_dir}/mono/decode_test

echo -e "\n-------------------------------------"
#cat ${output_dir}/mono/decode_train/scoring_kaldi/best_wer
cat ${output_dir}/mono/decode_dev/scoring_kaldi/best_wer
cat ${output_dir}/num_iters${num_iters}/totgauss${totgauss}/mono/decode_test/scoring_kaldi/best_wer
echo -e "-------------------------------------\n"
