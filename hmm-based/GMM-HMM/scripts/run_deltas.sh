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
mono_ali=$3
totgauss=$4
numleaves=$5
lang_dir=$6
output_dir=$7/totgauss${totgauss}/numleaves${numleaves}

steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" $numleaves $totgauss ${data_dir}/${train} ${lang_dir} ${mono_ali} ${output_dir}/tri1 || exit 1;

utils/mkgraph.sh ${lang_dir} ${output_dir}/tri1 ${output_dir}/tri1/graph
#local/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" ${output_dir}/tri1/graph ${data_dir}/train ${output_dir}/tri1/decode_train
local/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" ${output_dir}/tri1/graph ${data_dir}/dev ${output_dir}/tri1/decode_dev
local/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" ${output_dir}/tri1/graph ${data_dir}/test ${output_dir}/tri1/decode_test

echo -e "\n-------------------------------------"
#cat ${output_dir}/tri1/decode_train/scoring_kaldi/best_wer
cat ${output_dir}/tri1/decode_dev/scoring_kaldi/best_wer
cat ${output_dir}/tri1/decode_test/scoring_kaldi/best_wer
echo -e "-------------------------------------\n"

