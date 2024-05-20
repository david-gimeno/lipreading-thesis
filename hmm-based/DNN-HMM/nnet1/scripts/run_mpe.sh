#!/bin/bash

echo "Sequence Discriminative Training: Minimum Phone Error (MPE)"
echo ""

. ./path.sh
. ./cmd.sh
PATH=$PATH:/home/dgimeno/kaldi/src/nnetbin/:.

nj=10
data_fmllr_dir=$1
train=$2
lang_dir=$3
nnet_dir=$4
output_dir=$5

# Firs we generate lattices and alignments
steps/nnet/align.sh --nj $nj --cmd "$train_cmd" $data_fmllr_dir/$train $lang_dir $nnet_dir $output_dir/nnet_ali || exit 1;
steps/nnet/make_denlats.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.config \
    $data_fmllr_dir/$train $lang_dir $nnet_dir $output_dir/nnet_denlats || exit 1;

steps/nnet/train_mpe.sh --cmd "$train_cmd" --num-iters 15 --acwt 0.1 --do-smbr false \
    $data_fmllr_dir/$train $lang_dir $nnet_dir $output_dir/nnet_ali $output_dir/nnet_denlats $output_dir/nnet_mpe || exit 1;

for iter in {1..15}; do
  steps/nnet/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.config --nnet $output_dir/nnet_mpe/$iter.nnet \
     $nnet_dir/graph $data_fmllr_dir/test $output_dir/nnet_mpe/decode_test_${iter};
done;

echo -e "\n###########################"
for iter in {1..15}; do
  cat $output_dir/nnet_mpe/decode_test_$iter/scoring_kaldi/best_wer;
done;
echo -e "###########################"
