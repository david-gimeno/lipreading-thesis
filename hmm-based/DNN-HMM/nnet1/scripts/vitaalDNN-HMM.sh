#!/bin/bash

echo ""
echo "Let's start..."
echo ""

. ./path.sh || exit 1
. ./cmd.sh || exit 1

#export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64/
PATH=$PATH:/home/dgimeno/kaldi/src/nnetbin/:.
PATH=$PATH:/home/dgimeno/kaldi/src/nnet3bin/:.

nj=10
audiomodality=$1
modality="video"
gmmhmm=$2
destroot=$3

echo ""
echo "==== EXTRACTING ALIGNMENTS FROM SAT SYSTEM ===="
echo ""
rm -rf ${gmmhmm}_ali_train ${gmmhmm}_ali_dev ${gmmhmm}_ali_test

steps/align_fmllr.sh --nj $nj --cmd "run.pl" data/${audiomodality}/train data/lang $gmmhmm ${gmmhmm}_ali_train
steps/align_fmllr.sh --nj $nj --cmd "run.pl" data/${audiomodality}/dev data/lang $gmmhmm ${gmmhmm}_ali_dev
steps/align_fmllr.sh --nj $nj --cmd "run.pl" data/${audiomodality}/test data/lang $gmmhmm ${gmmhmm}_ali_test

echo ""
echo "==== OBTAINING fMLLR FEATURES  ===="
echo ""

# Adding this code line:
#
#    [ ! -f $srcdir/wavs.sc ] && validate_opts="$validate_opts --no-wav"
#
# to the script utils/copy_data_dir.sh

rm -rf data/fmllr-feats
data_fmllr=data/fmllr-feats
video_gmmhmm="./exp/LIP-RTVE/speaker-dependent/video/resnet_visual_features/100fps/topoA/context3/tri3b"

for dataset in train dev test; do
    dir=${data_fmllr}/${dataset}

    if [[ "$dataset" == "train" ]]; then
        tdir=${video_gmmhmm}_ali_train/
    else
        tdir=${video_gmmhmm}/decode_${dataset}
    fi
    steps/nnet/make_fmllr_feats.sh --nj $nj --cmd "$train_cmd" \
        --transform-dir $tdir \
        $dir data/${modality}/${dataset} $video_gmmhmm $dir/log $dir/data || exit 1; echo "";
done;
# split the data : 90% train 10% cross-validation (held-out)
#utils/subset_data_dir_tr_cv.sh data/fmllr-feats/train data/fmllr-feats/train_tr90 data/fmllr-feats/train_cv10 || exit 1;

echo
echo "===== PRE-TRAINING Deep Belief Network ====="
echo

for net in dnn; do #dnn cnn1d cnn2d lstm
for sp in 5; do

    rm -rf ${destroot}/pretrain-dbn
    steps/nnet/pretrain_dbn.sh --rbm-iter 3 --splice $sp data/fmllr-feats/train ${destroot}/pretrain-dbn || exit 1;

    echo
    echo "===== DNN-HMM Hybrid System TRAINING ====="
    echo "           Frame cross-entropy            "
    echo

    ali=${gmmhmm}_ali
    feature_transform=${destroot}/pretrain-dbn/final.feature_transform
    dbn=${destroot}/pretrain-dbn/6.dbn


    for actv in Sigmoid; do # Sigmoid Tanh ParametricRelu
    for hl in 4; do # 4
    for hd in 1024; do # 1024
    for lr in 0.008; do # 0.008
    for m in 0; do # 0
    for hf in 0.5; do # 0.5
    for klr in 0; do # 0

        #dir=exp/vitaal/dnnhmm/${net}/${actv}/nnet_hl${hl}_hd${hd}_lr${lr}_m${m}_hf${hf}_klr${klr}_${sp}
        dir=${destroot}/nnet2_vitaal
        steps/nnet/train.sh --proto-opts "--activation-type <"${actv}">" \
          --feature-transform $feature_transform --dbn $dbn \
          --scheduler-opts "--end-halving-impr 0.001 --halving-factor "${hf}" --momentum "${m}" --max-iters 20 --keep-lr-iters "${klr} \
          --hid-layers $hl --hid-dim $hd --learn-rate $lr --splice $sp \
          --network-type ${net} \
          $data_fmllr/train $data_fmllr/dev data/lang ${ali}_train ${ali}_dev ${dir} || exit 1;

        #utils/mkgraph.sh data/lang $dir ${dir}/graph || exit 1;
        #REUSING HCLG graph from GMM-HMM system
        steps/nnet/decode.sh --nj 10 --cmd "$decode_cmd" --config conf/decodeDNN.config --acwt 0.1 \
            ${gmmhmm}/graph data/fmllr-feats/test ${dir}/decode_test || exit 1;
        echo "";
        cat ${dir}/decode_test/scoring_kaldi/best_wer
        echo "";

        #steps/nnet/make_denlats.sh --nj 1 --cmd "run.pl" --config conf/decodeDNN.config --acwt $acwt  data/train data/lang $srcdir exp/nnet_denlats || exit 1;
done;
done;
done;
done;
done;
done;
done;
done;
done;
