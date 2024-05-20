#!/bin/bash

data_dir=$1

rm -rf ./FFTs/

for dataset in train dev test; do
    mkdir -p ./FFTs/${dataset}/
    compute-spectrogram-feats --frame-length=100 --frame-shift=40 scp:${data_dir}/${dataset}/wav.scp ark,scp:./FFTs/${dataset}/feats.ark,./FFTs/${dataset}/feats.scp
done;
