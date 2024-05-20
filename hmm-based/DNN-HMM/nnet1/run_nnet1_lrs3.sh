#!/bin/bash

echo ""
echo "Let's start..."
echo ""

. ./path.sh || exit 1
. ./cmd.sh || exit 1

PATH=$PATH:/home/dgimeno/kaldi/src/nnetbin/:.
PATH=$PATH:/home/dgimeno/kaldi/src/nnet3bin/:.

nj=10
data_dir=$1
train=$2
lang_dir=$3
gmmhmm_dir=$4
output_dir=$5

echo ""
echo "==== EXTRACTING ALIGNMENTS FROM THE GMM-HMM SYSTEM ===="
echo ""

steps/align_fmllr.sh --nj $nj --cmd "run.pl" ${data_dir}/${train} $lang_dir $gmmhmm_dir ${output_dir}/gmmhmm_ali_${train}
# steps/align_fmllr.sh --nj $nj --cmd "run.pl" ${data_dir}/dev $lang_dir $gmmhmm_dir ${output_dir}/gmmhmm_ali_dev

echo ""
echo "==== OBTAINING fMLLR FEATURES  ===="
echo ""

# Adding this code line:
#
#    [ ! -f $srcdir/wavs.scp ] && validate_opts="$validate_opts --no-wav"
#
# to the script utils/copy_data_dir.sh


data_fmllr_dir=./data/fmllr-feats/$train/
# if [ -d "$data_fmllr_dir" ]; then
#   rm -rf $data_fmllr_dir/
# else
#   mkdir -p $data_fmllr_dir
# fi

for dataset in $train test; do
    if [[ "$dataset" == "$train" ]]; then
        transform_dir=$output_dir/gmmhmm_ali_$train/
    else
        transform_dir=$gmmhmm_dir/decode_$dataset
    fi

    steps/nnet/make_fmllr_feats.sh --nj $nj --cmd "$train_cmd" \
      --transform-dir $transform_dir $data_fmllr_dir/$dataset \
      $data_dir/$dataset $gmmhmm_dir $data_fmllr_dir/$dataset/log $data_fmllr_dir/$dataset/data || exit 1; echo "";

    # split the data : 90% train 10% cross-validation (held-out)
    utils/subset_data_dir_tr_cv.sh $data_fmllr_dir/$train $data_fmllr_dir/${train}_tr90 $data_fmllr_dir/${train}_cv10 || exit 1;

done;

echo
echo "===== PRE-TRAINING Deep Belief Network ====="
echo

for net in dnn; do #dnn cnn1d cnn2d lstm
for sp in 5; do

  rm -rf ${output_dir}/pretrain-dbn
  steps/nnet/pretrain_dbn.sh --rbm-iter 3 --splice $sp $data_fmllr_dir/${train}_tr90 $output_dir/pretrain-dbn || exit 1;

  echo
  echo "===== DNN-HMM Hybrid System TRAINING ====="
  echo "           Frame cross-entropy            "
  echo

  ali_root=$output_dir/gmmhmm_ali
  feature_transform=$output_dir/pretrain-dbn/final.feature_transform
  dbn=$output_dir/pretrain-dbn/6.dbn


  for actv in Sigmoid; do # Sigmoid Tanh ParametricRelu
  for hl in 2; do # 0 2 4 6
  for hd in 1024; do # 256 512 1024
  for lr in 0.008; do # 0.008
  for m in 0; do # 0.1 0.3 0.5 0.7; do
  for hf in 0.5; do # 0.3 0.1; do
  for klr in 0; do # 5 10; do

    #dir=exp/vitaal/dnnhmm/${net}/${actv}/nnet_hl${hl}_hd${hd}_lr${lr}_m${m}_hf${hf}_klr${klr}_${sp}
    nnet_dir=$output_dir/nnet1/
    steps/nnet/train.sh --proto-opts "--activation-type <"${actv}">" \
      --feature-transform $feature_transform --dbn $dbn \
      --scheduler-opts "--end-halving-impr 0.001 --halving-factor "${hf}" --momentum "${m}" --max-iters 20 --keep-lr-iters "${klr} \
      --hid-layers $hl --hid-dim $hd --learn-rate $lr --splice $sp --network-type $net \
      $data_fmllr_dir/${train}_tr90 $data_fmllr_dir/${train}_cv10 $lang_dir ${ali_root}_${train} ${ali_root}_${train} $nnet_dir || exit 1;

    #utils/mkgraph.sh data/lang $dir ${dir}/graph || exit 1;
    #REUSING HCLG graph from GMM-HMM system
    steps/nnet/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.config --acwt 0.1 \
      $gmmhmm_dir/graph $data_fmllr_dir/test $nnet_dir/decode_test || exit 1;

    echo "";
    cat $nnet_dir/decode_test/scoring_kaldi/best_wer
    echo ""

  done;
  done;
  done;
  done;
  done;
  done;
  done;
done;
done;

./scripts/run_smbr.sh $data_fmllr_dir $train $lang_dir $nnet_dir $gmmhmm_dir $output_dir/nnet1+smbr/
