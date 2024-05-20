#!/bin/bash

echo -e ""
echo -e "Let's start..."
echo -e ""

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

nj=10

data_dir=$1
lang_dir=$2
ali_dir=$3
output_dir=$4

echo -e ""
echo -e "==== APPLYING THE ViTAAl STRATEGY FROM THE LDA+MLLT STAGE ===="
echo -e ""

steps/train_lda_mllt.sh --cmd "$train_cmd" --splice-opts "--left-context=3 --right-context=3" \
    --train-tree false --realign-iters "" 2500 15000 ${data_dir}/train $lang_dir $ali_dir ${output_dir}/tri2b || exit 1;

utils/mkgraph.sh $lang_dir ${output_dir}/tri2b ${output_dir}/tri2b/graph
for dataset in test; do
  local/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" ${output_dir}/tri2b/graph ${data_dir}/${dataset} ${output_dir}/tri2b/decode_${dataset}

  echo -e "\n-------------------------------------"
  cat ${output_dir}/tri2b/decode_${dataset}/scoring_kaldi/best_wer
  echo -e "-------------------------------------\n"
done;

echo -e ""
echo -e "==== BUILDING THE LDA+MLLT+SAT SYSTEM FROM THE ViTAAl-LDA+MLLT MODEL ===="
echo -e ""

steps/align_si.sh --nj $nj --cmd "$train_cmd" ${data_dir}/train ${lang_dir} ${output_dir}/tri2b ${output_dir}/tri2b_ali || exit 1;
steps/train_sat.sh --cmd "$train_cmd" --train-tree false 4200 40000 ${data_dir}/train ${lang_dir} ${output_dir}/tri2b_ali ${output_dir}/tri3b
steps/align_fmllr.sh --nj 10 --cmd "$train_cmd" ${data_dir}/train $lang_dir ${output_dir}/tri3b ${output_dir}/tri3b_ali

utils/mkgraph.sh ${lang_dir} ${output_dir}/tri3b ${output_dir}/tri3b/graph
for dataset in test; do
  local/decode_fmllr.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" ${output_dir}/tri3b/graph ${data_dir}/${dataset} ${output_dir}/tri3b/decode_${dataset}

  echo -e "\n-------------------------------------"
  cat ${output_dir}/tri3b/decode_${dataset}/scoring_kaldi/best_wer
  echo -e "-------------------------------------\n"
done;
