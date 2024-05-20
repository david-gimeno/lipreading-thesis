#!/bin/bash

for training_data in 1hour 2hours 5hours 10hours 20hours 50hours 100hours fulltrain; do
#   ./run.sh ../data/LRS2-BBC/25fps/speaker-independent/conformer-from-lrs2/ ${training_data} \
#            ../data/LRS2-BBC/25fps/speaker-independent/lm_data/topoB/word/3gram/lang/ \
#            ./exp/LRS2-BBC/25fps/speaker-independent/topoB/word/3gram/conformer-from-lrs2/${training_data}/ > lrs2_topoB_word3gram_conformer-from-lrs2_${training_data}.out
  ./run.sh ../data/LRS2-BBC/100fps/speaker-independent/PASE+/ ${training_data} \
           ../data/LRS2-BBC/100fps/speaker-independent/lm_data/topoA/word/3gram/lang/ \
           ./exp/LRS2-BBC/100fps/speaker-independent/topoA/word/3gram/PASE+/${training_data}/ > lrs2_topoA_word3gram_PASE+_${training_data}.out
# for training_data in 1hour 2hours 5hours 10hours 20hours 50hours 100hours 200hours fulltrain; do
#   ./run.sh ../data/LRS3-TED/25fps/speaker-independent/conformer-from-lrs2-finetuned-to-lrs3/ ${training_data} \
#            ../data/LRS3-TED/25fps/speaker-independent/lm_data/topoB/word/3gram/lang/ \
#            ./exp/LRS3-TED/25fps/speaker-independent/topoB/word/3gram/conformer-from-lrs2-finetuned-to-lrs3/${training_data}/ > lrs3_topoB_word3gram_conformer-from-lrs2-finetuned-to-lrs3_${training_data}.out
done;
