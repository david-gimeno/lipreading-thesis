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
tri2b_ali=$3
totgauss=$4
numleaves=$5
lang_dir=$6
output_dir=$7/totgauss${totgauss}/numleaves${numleaves}

echo -e ""
echo -e "==== TRAINING LDA+MLLT+SAT (fMLLR) SYSTEM  ===="
echo -e ""

steps/train_sat.sh --cmd "$train_cmd" $numleaves $totgauss ${data_dir}/${train} ${lang_dir} ${tri2b_ali} ${output_dir}/tri3b

steps/align_fmllr.sh --nj 10 --cmd "$train_cmd" ${data_dir}/${train} $lang_dir ${output_dir}/tri3b ${output_dir}/tri3b_ali

utils/mkgraph.sh ${lang_dir} ${output_dir}/tri3b ${output_dir}/tri3b/graph
#local/decode_fmllr.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" ${output_dir}/tri3b/graph ${data_dir}/train ${output_dir}/tri3b/decode_train
local/decode_fmllr.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" ${output_dir}/tri3b/graph ${data_dir}/dev ${output_dir}/tri3b/decode_dev
local/decode_fmllr.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" ${output_dir}/tri3b/graph ${data_dir}/test ${output_dir}/tri3b/decode_test

echo -e "\n-------------------------------------"
#cat ${output_dir}/tri3b/decode_train/scoring_kaldi/best_wer
cat ${output_dir}/tri3b/decode_dev/scoring_kaldi/best_wer
cat ${output_dir}/tri3b/decode_test/scoring_kaldi/best_wer
echo -e "-------------------------------------\n"

