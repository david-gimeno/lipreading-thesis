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
tri1_ali=$3
dim=$4
context=$5
totgauss=$6
numleaves=$7
lang_dir=$8
output_dir=$9/dim${dim}/context${context}/totgauss${totgauss}/numleaves${numleaves}


echo -e ""
echo -e "==== TRAINING LDA+MLLT SYSTEM  ===="
echo -e ""

steps/train_lda_mllt.sh --cmd "$train_cmd" --dim $dim --splice-opts "--left-context="${context}" --right-context="${context} $numleaves $totgauss \
    ${data_dir}/${train} ${lang_dir} ${tri1_ali} ${output_dir}/tri2b || exit 1;

utils/mkgraph.sh ${lang_dir} ${output_dir}/tri2b ${output_dir}/tri2b/graph
#local/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" ${output_dir}/tri2b/graph ${data_dir}/train ${output_dir}/tri2b/decode_train
local/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" ${output_dir}/tri2b/graph ${data_dir}/dev ${output_dir}/tri2b/decode_dev
local/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" ${output_dir}/tri2b/graph ${data_dir}/test ${output_dir}/tri2b/decode_test

echo -e "\n-------------------------------------"
#cat ${output_dir}/tri2b/decode_train/scoring_kaldi/best_wer
cat ${output_dir}/tri2b/decode_dev/scoring_kaldi/best_wer
cat ${output_dir}/tri2b/decode_test/scoring_kaldi/best_wer
echo -e "-------------------------------------\n"

