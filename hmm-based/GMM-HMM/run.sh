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
lang_dir=$3
output_dir=$4

echo -e ""
echo -e "==== TRAINING MONOPHONE SYSTEM ===="
echo -e ""

steps/train_mono.sh --nj $nj --cmd "$train_cmd" --boost-silence 1.25 ${data_dir}/${train} ${lang_dir} ${output_dir}/mono || exit 1;

utils/mkgraph.sh ${lang_dir} ${output_dir}/mono ${output_dir}/mono/graph
#local/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" ${output_dir}/mono/graph ${data_dir}/train ${output_dir}/mono/decode_train
local/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" ${output_dir}/mono/graph ${data_dir}/dev ${output_dir}/mono/decode_dev
local/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" ${output_dir}/mono/graph ${data_dir}/test ${output_dir}/mono/decode_test

echo -e "\n-------------------------------------"
#cat ${output_dir}/mono/decode_train/scoring_kaldi/best_wer
cat ${output_dir}/mono/decode_dev/scoring_kaldi/best_wer
cat ${output_dir}/mono/decode_test/scoring_kaldi/best_wer
echo -e "-------------------------------------\n"

echo -e ""
echo -e "==== TRAINING DELTAS SYSTEM  ===="
echo -e ""

steps/align_si.sh --nj $nj --cmd "$train_cmd" --boost-silence 1.25 ${data_dir}/${train} ${lang_dir} ${output_dir}/mono ${output_dir}/mono_ali || exit 1;
steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2000 10000 ${data_dir}/${train} ${lang_dir} ${output_dir}/mono_ali ${output_dir}/tri1 || exit 1;

utils/mkgraph.sh ${lang_dir} ${output_dir}/tri1 ${output_dir}/tri1/graph
#local/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" ${output_dir}/tri1/graph ${data_dir}/train ${output_dir}/tri1/decode_train
local/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" ${output_dir}/tri1/graph ${data_dir}/dev ${output_dir}/tri1/decode_dev
local/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" ${output_dir}/tri1/graph ${data_dir}/test ${output_dir}/tri1/decode_test

echo -e "\n-------------------------------------"
#cat ${output_dir}/tri1/decode_train/scoring_kaldi/best_wer
cat ${output_dir}/tri1/decode_dev/scoring_kaldi/best_wer
cat ${output_dir}/tri1/decode_test/scoring_kaldi/best_wer
echo -e "-------------------------------------\n"


echo -e ""
echo -e "==== TRAINING LDA+MLLT SYSTEM  ===="
echo -e ""

steps/align_si.sh --nj $nj --cmd "$train_cmd" ${data_dir}/${train} ${lang_dir} ${output_dir}/tri1 ${output_dir}/tri1_ali || exit 1;
steps/train_lda_mllt.sh --cmd "$train_cmd" --splice-opts "--left-context=3 --right-context=3" 2500 15000 ${data_dir}/${train} ${lang_dir} ${output_dir}/tri1_ali ${output_dir}/tri2b || exit 1;

utils/mkgraph.sh ${lang_dir} ${output_dir}/tri2b ${output_dir}/tri2b/graph
#local/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" ${output_dir}/tri2b/graph ${data_dir}/train ${output_dir}/tri2b/decode_train
local/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" ${output_dir}/tri2b/graph ${data_dir}/dev ${output_dir}/tri2b/decode_dev
local/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" ${output_dir}/tri2b/graph ${data_dir}/test ${output_dir}/tri2b/decode_test

echo -e "\n-------------------------------------"
#cat ${output_dir}/tri2b/decode_train/scoring_kaldi/best_wer
cat ${output_dir}/tri2b/decode_dev/scoring_kaldi/best_wer
cat ${output_dir}/tri2b/decode_test/scoring_kaldi/best_wer
echo -e "-------------------------------------\n"

echo -e ""
echo -e "==== TRAINING LDA+MLLT+SAT (fMLLR) SYSTEM  ===="
echo -e ""

steps/align_si.sh --nj $nj --cmd "$train_cmd" ${data_dir}/${train} ${lang_dir} ${output_dir}/tri2b ${output_dir}/tri2b_ali || exit 1;
steps/train_sat.sh --cmd "$train_cmd" 4200 40000 ${data_dir}/${train} ${lang_dir} ${output_dir}/tri2b_ali ${output_dir}/tri3b

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

