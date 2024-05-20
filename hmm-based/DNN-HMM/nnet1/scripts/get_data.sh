#!/bin/bash

gmmhmm_dir=../../GMM-HMM/exp/LRS2-BBC/25fps/speaker-independent/topoB/word/3gram/conformer/
output_dir=exp/LRS2-BBC/25fps/speaker-independent/topoB/word/3gram/conformer/

data_fmllr_dir=./data/fmllr-feats
if [ -d "$data_fmllr_dir" ]; then
  rm -rf $data_fmllr_dir
fi
mkdir -p $data_fmllr_dir

for dataset in 1hour 2hours 5hours 10hours 20hours 50hours 100hours fulltrain dev test; do
    train_ali_dir=$output_dir/$dataset/gmmhmm_ali_$dataset
    if [[ "$dataset" == "$train" ]]; then
        transform_dir=$train_ali_dir
    else
        transform_dir=$gmmhmm_dir/$dataset/tri3b/decode_$dataset
    fi

    steps/nnet/make_fmllr_feats.sh --nj $nj --cmd "$train_cmd" \
      --transform-dir $transform_dir $data_fmllr_dir/$dataset \
      $data_dir/$dataset $gmmhmm_dir $data_fmllr_dir/$dataset/log $data_fmllr_dir/$dataset/data || exit 1; echo "";
done;
