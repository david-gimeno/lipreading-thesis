#!/bin/bash

database="LRS2-BBC"
scenario="speaker-independent"
modality="conformer"
train="10hours"
fps="25fps"

mono_ali=$1

data_dir=../data/${database}/${fps}/${scenario}/${modality}/
lang_dir=../data/${database}/${fps}/${scenario}/lm_data/topoB/word/3gram/lang/
output_dir=./exp-deltas/${database}/${fps}/${scenario}/${modality}/${train}/topoB/word/3gram/

## Fixing totgauss=25000 ## Exploring numleaves: 1000 1500 2000 2500 3000 3500 4000 4500 5000 ## No significant difference --> 2000
  ## Fixing numleaves=2000 ## Exploring totgauss: 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 ## No significant difference --> 10000

for totgauss in 10000; do
for numleaves in 2000; do
  dst_dir=./outputs-exp-deltas/${database}/${fps}/${scenario}/${train}/topoB/word/3gram/totgauss${totgauss}/numleaves${numleaves}/
  mkdir -p $dst_dir
  ./scripts/run_deltas.sh $data_dir $train $mono_ali $totgauss $numleaves $lang_dir $output_dir > ${dst_dir}/log.out

done;
done;
