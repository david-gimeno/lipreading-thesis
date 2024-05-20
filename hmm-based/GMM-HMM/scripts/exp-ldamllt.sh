#!/bin/bash

database="LRS2-BBC"
scenario="speaker-independent"
modality="conformer"
train="10hours"
fps="25fps"

tri1_ali=$1

data_dir=../data/${database}/${fps}/${scenario}/${modality}/
lang_dir=../data/${database}/${fps}/${scenario}/lm_data/topoB/word/3gram/lang/
output_dir=./exp-ldamllt/${database}/${fps}/${scenario}/${modality}/${train}/topoB/word/3gram/

## Fixing dim=40, context=3, totgauss=45000 ## Exploring numleaves: 1500 2500 3500 4500 5500 6500 7500 ## No signficant differences --> 2500
## Fixing dim=40, context=3, numleaves=2500 ## Exploring numleaves: 15000 25000 45000 75000 ## No signficant differences --> 15000
## Fixing totgauss=15000 ## Exploring dim: 40 64 80 128 256 & context: 1 2 3 4 5 6 7 ## No significant differences --> 40, 3


for dim in 40; do
for context in 3; do
for totgauss in 15000; do
for numleaves in 2500; do

  dst_dir=./outputs-exp-ldamllt/${database}/${fps}/${scenario}/${train}/topoB/word/3gram/dim${dim}/context${context}/totgauss${totgauss}/numleaves${numleaves}/
  mkdir -p $dst_dir
  ./scripts/run_ldamllt.sh $data_dir $train $tri1_ali $dim $context $totgauss $numleaves $lang_dir $output_dir > ${dst_dir}/log.out

done;
done;
done;
done;
