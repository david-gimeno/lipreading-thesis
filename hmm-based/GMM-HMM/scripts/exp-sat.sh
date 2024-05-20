#!/bin/bash

database="LRS2-BBC"
scenario="speaker-independent"
modality="conformer"
train="10hours"
fps="25fps"

tri2b_ali=$1

data_dir=../data/${database}/${fps}/${scenario}/${modality}/
lang_dir=../data/${database}/${fps}/${scenario}/lm_data/topoB/word/3gram/lang/
output_dir=./exp-sat/${database}/${fps}/${scenario}/${modality}/${train}/topoB/word/3gram/

# DEFAULT: 4200 40000
# Exploring:
# totgauss --> 15000 20000 40000 60000 120000 200000
# numleaves --> 2500 4200 5000 7500 10000
for totgauss in 15000; do
for numleaves in 2500; do

  dst_dir=./outputs-exp-sat/${database}/${fps}/${scenario}/${train}/topoB/word/3gram/totgauss${totgauss}/numleaves${numleaves}/
  mkdir -p $dst_dir
  ./scripts/run_sat.sh $data_dir $train $tri2b_ali $totgauss $numleaves $lang_dir $output_dir > ${dst_dir}/log.out

done;
done;
