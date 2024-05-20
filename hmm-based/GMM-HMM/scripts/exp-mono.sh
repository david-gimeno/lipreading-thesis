#!/bin/bash

database="LRS2-BBC"
scenario="speaker-independent"
modality="conformer"
train="10hours"
fps="25fps"

lang_dir=../data/${database}/${fps}/${scenario}/lm_data/topoB/word/3gram/lang/
output_dir=./exp-mono/${database}/${fps}/${scenario}/${modality}/${train}/topoB/word/3gram/


## totgauss: 800 1000 1200 1400 1600 1800 2000 10000 ## No significant difference  ## LRS2-BBC
## num_iters: 30 35 40 45 50 ## No significant difference ## LRS2-BBC

for num_iters in 40; do
  for totgauss in 1000; do
    dst_dir=./outputs-exp-mono/${database}/${fps}/${scenario}/${train}/topoB/word/3gram/numiters${num_iters}/totgauss${totgauss}
    mkdir -p $dst_dir

    ./scripts/run_mono.sh ../data/${database}/${fps}/${scenario}/${modality}/ $train \
                          $num_iters $totgauss $lang_dir $output_dir > $dst_dir/log.out
  done
done
