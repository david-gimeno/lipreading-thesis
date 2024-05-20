#!/bin/bash


for dataset in test; do
for training_data in fulltrain; do # 100hours 200hours fulltrain; do
  ## DNN-HMM+sMBR (nnet1) ##
  # ./lm-rescoring.sh 100 \
  #   ../DNN-HMM/nnet1/exp/LRS2-BBC/25fps/speaker-independent/topoB/word/3gram/conformer-from-lrs2/$training_data/nnet1+smbr/nnet_smbr/decode_${dataset}_5/ \
  #   ../data/LRS2-BBC/25fps/speaker-independent/lm_data/topoB/word/3gram/lang/ \
  #   ../DNN-HMM/nnet1/data/fmllr-feats/$training_data/$dataset/ \
  #   ../DNN-HMM/nnet1/exp/LRS2-BBC/25fps/speaker-independent/topoB/character/transformerLM/conformer-from-lrs2/$training_data/nnet1+smbr/nnet_smbr/decode_${dataset}_5/ \
  #   LRS2-BBC 0.1 18 > lrs2_dnn-hmm_lm-rescoring_${training_data}.out

  ./lm-rescoring.sh 100 \
    ../DNN-HMM/nnet1/exp/LRS3-TED/25fps/speaker-independent/topoB/word/3gram/conformer-from-lrs2-finetuned-to-lrs3/$training_data/nnet1+smbr/nnet_smbr/decode_${dataset}_5/ \
    ../data/LRS3-TED/25fps/speaker-independent/lm_data/topoB/word/3gram/lang/ \
    ../DNN-HMM/nnet1/data/fmllr-feats/$training_data/$dataset/ \
    ../DNN-HMM/nnet1/exp/LRS3-TED/25fps/speaker-independent/topoB/character/transformerLM/conformer-from-lrs2-finetuned-to-lrs3/$training_data/nnet1+smbr/nnet_smbr/decode_${dataset}_5/ \
    LRS3-TED 0.1 18 > lrs3_dnn-hmm_lm-rescoring_${training_data}.out


  ## GMM-HMM ##
  # ./lm-rescoring.sh 100 \
  #   ../GMM-HMM/exp/LRS2-BBC/25fps/speaker-independent/topoB/word/3gram/conformer-from-lrs2/$training_data/tri3b/decode_${dataset}/ \
  #   ../data/LRS2-BBC/25fps/speaker-independent/lm_data/topoB/word/3gram/lang/ \
  #   ../data/LRS2-BBC/25fps/speaker-independent/conformer-from-lrs2/$dataset/ \
  #   ../GMM-HMM/exp/LRS2-BBC/25fps/speaker-independent/topoB/character/transformerLM/conformer-from-lrs2/$training_data/tri3b/decode_${dataset}/ \
  #   LRS2-BBC 0.083333 13 > lrs2_gmm-hmm_lm-rescoring_${training_data}.out
done;
done;
