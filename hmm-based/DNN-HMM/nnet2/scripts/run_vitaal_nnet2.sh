#!/usr/bin/env bash

# This is pnorm neural net training on top of adapted 40-dimensional features.


train_stage=-10
use_gpu=true

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

#ln -s ~/kaldi/egs/wsj/s5/utils utils
#ln -s ~/kaldi/egs/wsj/s5/steps steps

data_dir=$1
lang_dir=$2
gmm_dir=$3
acoustic_ali_dir=$4
root_output_dir=$5

if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
  fi
  parallel_opts="--gpu 1"
  num_threads=1
  minibatch_size=256
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  parallel_opts="--num-threads $num_threads"
  minibatch_size=128
fi

for hl in 2; do
for i in 2000,400; do
for lr in 0.02; do
  IFS="," read input_dim output_dim <<< "${i}"

  output_dir=${root_output_dir}_hl${hl}_lr_${lr}_dim${input_dim}-${output_dim}
  steps/nnet2/train_pnorm_bottleneck_fast.sh --stage $train_stage \
   --samples-per-iter 200000 \
   --parallel-opts "$parallel_opts" \
   --num-threads "$num_threads" \
   --minibatch-size "$minibatch_size" \
   --num-jobs-nnet 8  --mix-up 8000 \
   --initial-learning-rate $lr --final-learning-rate 0.001 \
   --num-hidden-layers $hl \
   --pnorm-input-dim $input_dim --pnorm-output-dim $output_dim \
   --cmd "$decode_cmd" \
   ${data_dir}/train $lang_dir $acoustic_ali_dir ${output_dir} || exit 1

  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 10 --config ./conf/decode.config \
   --transform-dir ${gmm_dir}/decode_test/ \
   ${gmm_dir}/graph ${data_dir}/test ${output_dir}/decode_test

  echo -e "\n-------------------------------------"
  cat ${output_dir}/decode_test/scoring_kaldi/best_wer
  echo -e "-------------------------------------\n"

done;
done;
done;
