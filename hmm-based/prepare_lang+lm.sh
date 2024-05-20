#!/bin/bash

# ./prepare_lang+lm.sh LRS2-BBC 100fps speaker-independent A character 7

DEFAULT="no-specified"

. ./path.sh
. ./cmd.sh

database=$1
modality=$2
fps=$3
scenario=$4
topology=$5
lm_level=$6
ngram=$7
text_path="${8:-$DEFAULT}"

dst_root=./data/${database}/${fps}/${scenario}

if [[ "$text_path" == *"no-specified"* ]]
then
  lmdata_dir=${dst_root}/lm_data/topo${topology}/${lm_level}/${ngram}gram
else
  from=$(echo $text_path | rev | cut -d/ -f1 | rev)
  lmdata_dir=${dst_root}/lm_data/from_${from}/topo${topology}/${lm_level}/${ngram}gram
fi

if [ -d "$lmdata_dir" ]; then
  rm -rf $lmdata_dir
fi
mkdir -p $lmdata_dir

echo -e "\n#################"
echo -e "PREPARING LM DATA"
echo -e "#################"


if [[ "$text_path" == *"no-specified"* ]]
then

  if [[ "$database" == *"LRS"* ]]
  then
    split_path=../data/${database}/splits/${scenario}/fulltrain${database}.csv
  else
    split_path=../data/${database}/splits/${scenario}/train${database}.csv
  fi
  python3 ./scripts/lm_data/get_lm_data_from_split.py $lm_level $split_path $lmdata_dir
else
  echo $text_path
  python3 ./scripts/lm_data/get_lm_data_from_txt.py $lm_level $text_path $lmdata_dir
fi

lang_dir=${lmdata_dir}/lang

local_dir=${lmdata_dir}/local
lexicon_dir=${local_dir}/dict
local_lang_dir=${local_dir}/lang

echo -e "\n######################"
echo -e "BUILDING LEXICON MODEL"
echo -e "######################"

python3 ./scripts/lm_data/get_lexical_data.py ${lmdata_dir}/train_lm.txt $lexicon_dir $lm_level
scripts/lm_data/prepare_lang.sh $lexicon_dir "<UNK>" $local_lang_dir $lang_dir $topology || exit 1;

echo -e "\n#######################"
echo -e "BUILDING LANGUAGE MODEL"
echo -e "#######################"

sdir=/home/dgimeno/SRILM-1.7.3/bin/i686-m64/
export PATH=$PATH:$sdir

tmp_dir=${local_dir}/tmp
if [ -d "$tmp_dir" ]; then
  rm -rf $tmp_dir
fi
mkdir -p $tmp_dir

ngram-count -text ${lmdata_dir}/train_lm.txt -order $ngram -wbdiscount -interpolate -lm $tmp_dir/lm.arpa; echo "";

ppl_dir=./data/${database}/${fps}/${scenario}/${modality}/
python3 scripts/lm_data/get_ppl_txts.py $ppl_dir $lm_level
for dataset in $(ls $ppl_dir); do
    if [[ ! "$dataset" == *"source-data"*  ]]; then
        # cut -f 2- -d ' ' ${ppl_dir}/${dataset}/text | gzip -c > ${ppl_dir}/${dataset}.textLM.gz
      cat ./${dataset}.textLM.txt | gzip -c > ./${dataset}.textLM.gz
      ngram -order $ngram -lm $tmp_dir/lm.arpa -ppl ./${dataset}.textLM.gz; echo "";
      rm ./${dataset}.textLM.txt ./${dataset}.textLM.gz
    fi
done;

arpa2fst --disambig-symbol=#0 --read-symbol-table=$lang_dir/words.txt $tmp_dir/lm.arpa $lang_dir/G.fst

