#!/bin/bash

# ./prepare_data.sh ../data/LRS2-BBC/WAVs/ audio speaker-independent 100fps

#rm -rf ./utils; ln -s ~/kaldi/egs/wsj/s5/utils utils
#rm -rf ./steps/; ln -s ~/kaldi/egs/wsj/s5/steps steps

. ./path.sh
. ./cmd.sh

pathdata=$1
modality=$2
scenario=$3
fps=$4

database=$(echo $pathdata | cut -f 3 -d '/')
rootdata=./data/${database}/${fps}/${scenario}/${modality}/
rootsource=${rootdata}/source-data/

# rm -rf ./data/${database}/${fps}/${scenario}/${modality}/

echo -e "\n########################"
echo "PREPARING DATA FOR KALDI"
echo -e "########################\n"

if [ ${scenario} == "no-scenario"  ]
then
    python3 ./scripts/input_data/get_splits.py $pathdata $database $modality $rootsource ""
else
    python3 ./scripts/input_data/get_splits.py $pathdata $database $modality $rootsource $scenario
fi

python3 ./scripts/input_data/get_text.py $rootsource $rootdata $database
python3 ./scripts/input_data/get_utt2spk.py $rootsource $rootdata $database

for dataset in $(ls $rootsource); do
    utils/utt2spk_to_spk2utt.pl ${rootdata}/${dataset}/utt2spk > ${rootdata}/${dataset}/spk2utt || exit 1;
done;

if [[ "$modality" == *"audio"* ]]
then
    python3 ./scripts/input_data/get_scp_wavs.py $rootsource $rootdata
else
    if [[ "$pathdata" == *"geometricFeats"*  ]]
    then
        numFeats=19
    elif [[ "$pathdata" == *"eigenlips"* ]]
    then
        numFeats=32
    elif [[ "$pathdata" == *"deepFeats"* ]]
    then
        numFeats=32
    elif [[ "$pathdata" == *"allFeats"* ]]
    then
        numFeats=83
    elif [[ "$pathdata" == *"resnet"*  ]]
    then
        numFeats=512
    elif [[ "$pathdata" == *"conformer"*  ]]
    then
        numFeats=256
    elif [[ "$pathdata" == *"fake-mfcc"*  ]]
    then
        numFeats=13
    elif [[ "$pathdata" == *"PASE"*  ]]
    then
        numFeats=256
    else
        echo "Unexpected visual speech features"
    fi

    python3 ./scripts/input_data/get_feats.py $rootsource $rootdata $numFeats $fps
fi

if [[ "$fps" == *"100"* ]]
then
    mfcc_config=./conf/mfcc100fps.conf
elif [[ "$fps" == *"50"* ]]
then
    mfcc_config=./conf/mfcc50fps.conf
else
    mfcc_config=./conf/mfcc25fps.conf
fi

for dataset in $(ls $rootsource); do
    if [ ! -d ${rootdata}/${dataset}/data/ ]
    then
      if [[ "$modality" == *"audio"* ]]
      then
          steps/make_mfcc.sh --cmd "$train_cmd" --mfcc-config $mfcc_config --nj 10 ${rootdata}/${dataset}/ || exit 1;
      fi
      steps/compute_cmvn_stats.sh ${rootdata}/${dataset}/ || exit 1;
    else
      echo "Dataset "${dataset}" is already processed"
    fi
done

# if [[ "$rootsource" == *"LIP-RTVE"* ]]
# then
#     utils/subset_data_dir.sh --shortest ${rootdata}/train 2000 ${rootdata}/train_mono
# elif [[ "$rootsource" == *"LRS3-TED"* ]]
# then
#     utils/subset_data_dir.sh --shortest ${rootdata}/train 7500 ${rootdata}/train_mono
# fi
if [[ "$modality" == *"audio"* ]]
then
  echo -e "\n##################"
  echo "WAV FILES COMPLETED"
  echo -e "###################\n"
fi
