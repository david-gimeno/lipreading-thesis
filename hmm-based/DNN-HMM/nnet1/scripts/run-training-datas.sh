#!/bin/bash

#for traindata in 1hour 2hours 5hours 10hours 20hours 50hours 100hours fulltrain; do
#   ./run_nnet1.sh ../../data/LRS2-BBC/25fps/speaker-independent/conformer-from-lrs2/ $traindata \
#     ../../data/LRS2-BBC/25fps/speaker-independent/lm_data/topoB/word/3gram/lang/ \
#     ../../GMM-HMM/exp/LRS2-BBC/25fps/speaker-independent/topoB/word/3gram/conformer-from-lrs2/$traindata/tri3b/ \
#     ./exp/LRS2-BBC/25fps/speaker-independent/topoB/word/3gram/conformer-from-lrs2/$traindata/ > lrs2_nnet1_$traindata.out

for traindata in 1hour 2hours 5hours 10hours 20hours 50hours 100hours 200hours fulltrain; do
  ./run_nnet1_lrs3.sh ../../data/LRS3-TED/25fps/speaker-independent/conformer-from-lrs2-finetuned-to-lrs3/ $traindata \
    ../../data/LRS3-TED/25fps/speaker-independent/lm_data/topoB/word/3gram/lang/ \
    ../../GMM-HMM/exp/LRS3-TED/25fps/speaker-independent/topoB/word/3gram/conformer-from-lrs2-finetuned-to-lrs3/$traindata/tri3b/ \
    ./exp/LRS3-TED/25fps/speaker-independent/topoB/word/3gram/conformer-from-lrs2-finetuned-to-lrs3/$traindata/ > lrs3_nnet1_$traindata.out
done;
