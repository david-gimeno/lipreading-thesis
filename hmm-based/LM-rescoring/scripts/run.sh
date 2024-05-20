#!/bin/bash

# ./lm-rescoring.sh 100 ../GMM-HMM/exp/LRS2-BBC/25fps/speaker-independent/topoB/word/3gram/conformer/100hours/tri3b/decode_test/ ../data/LRS2-BBC/25fps/speaker-independent/lm_data/topoB/word/3gram/lang/ ../data/LRS2-BBC/25fps/speaker-independent/conformer/test/ ../GMM-HMM/exp/LRS2-BBC/25fps/speaker-independent/topoB/transformerLM/conformer/100hours/decode_test/

nbest=100
dataset="test"
modality="conformer"
root_dir="../GMM-HMM/exp/LRS2-BBC/25fps/speaker-independent/topoB/"

exp_dir=${root_dir}"word/3gram/"${modality}"/"
lang_dir="../data/LRS2-BBC/25fps/speaker-independent/lm_data/topoB/word/3gram/lang/"
data_dir="../data/LRS2-BBC/25fps/speaker-independent/"${modality}"/"${dataset}"/"

for training_data in 1hour 2hours 5hours 10hours 20hours 50hours 100hours fulltrain; do
  decode_dir=${exp_dir}/${training_data}/tri3b/decode_${dataset}
  output_dir=${root_dir}/transformerLM/${modality}/${training_data}/decode_${dataset}/

  ./lm-rescoring.sh $nbest $decode_dir $lang_dir $data_dir $output_dir
done;
