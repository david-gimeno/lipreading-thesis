#!/bin/bash

. ./path.sh
. ./cmd.sh

nj=10

gmmdir=$1
data_dir=$2
dest_dir=$3

mkdir -p $dest_dir

# Adding this code line:
#
#    [ ! -f $srcdir/wavs.sc ] && validate_opts="$validate_opts --no-wav"
#
# to the script utils/copy_data_dir.sh

steps/nnet/make_fmllr_feats.sh --nj $nj --cmd "$train_cmd" \
  --transform-dir ${gmmdir}_ali \
  $dest_dir/train/ $data_dir/train/ $gmmdir $dest_dir/train/log $dest_dir/train/data || exit 1;

steps/nnet/make_fmllr_feats.sh --nj $nj --cmd "$train_cmd" \
  --transform-dir $gmmdir/decode_dev \
  $dest_dir/dev/ $data_dir/dev/ $gmmdir $dest_dir/dev/log $dest_dir/dev/data || exit 1;

steps/nnet/make_fmllr_feats.sh --nj $nj --cmd "$train_cmd" \
  --transform-dir $gmmdir/decode_test \
  $dest_dir/test/ $data_dir/test/ $gmmdir $dest_dir/test/log $dest_dir/test/data || exit 1;
