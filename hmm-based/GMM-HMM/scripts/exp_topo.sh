#!/bin/bash

database=$1
scenario=$2
modality=$3
fps=$4

for topo in A B B_v1 B_v2 C D E chain; do
  dst_dir=./topo_outputs/${database}/${fps}/${scenario}/20hours/topo${topo}/word/3gram/
  mkdir -p $dst_dir

  ./run.sh ../data/${database}/${fps}/${scenario}/${modality}/ 20hours \
           ../data/${database}/${fps}/${scenario}/lm_data/topo${topo}/word/3gram/lang/ \
           ./exp/${database}/${fps}/${scenario}/${modality}/20hours/topo${topo}/word/3gram/ > ${dst_dir}/log.out
done
