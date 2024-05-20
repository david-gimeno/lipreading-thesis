#!/bin/bash

# ./lm-rescoring.sh 100 ../GMM-HMM/exp/LRS2-BBC/25fps/speaker-independent/topoB/word/3gram/conformer/10hours/tri3b/decode_test/ ../data/LRS2-BBC/25fps/speaker-independent/lm_data/topoB/word/3gram/lang/ ../data/LRS2-BBC/25fps/speaker-independent/conformer/test/ ../GMM-HMM/exp/LRS2-BBC/25fps/speaker-independent/topoB/transformerLM/conformer/10hours/decode_test/ LRS2-BBC 0.083333

. ./cmd.sh
. ./path.sh

N=$1
lattice_dir=$2
lang_dir=$3
data_dir=$4
output_dir=$5

database=$6 # $(echo $lattice_dir | cut -f 5 -d '/')
acwt=$7 # 0.083333
beam=$8

nj=`cat $lattice_dir/num_jobs`

# if [ -d "$output_dir" ]; then
#   rm -rf $output_dir
# fi
mkdir -p $output_dir/nbest/

## CONVERTING LATTICE TO N-BEST ##
$cmd JOB=1:$nj $output_dir/log/lat2nbest.JOB.log \
  lattice-to-nbest --n=$N --acoustic-scale=$acwt \
  "ark:gunzip -c $lattice_dir/lat.JOB.gz|" \
  "ark:|gzip -c >$output_dir/nbest/nbest.JOB.gz" || exit 1;

## REMOVING OLD LMs SCORES ##
mkdir -p $output_dir/nbest_nolm/

$cmd JOB=1:$nj $output_dir/log/removeoldlm.JOB.log \
      gunzip -c $output_dir/nbest/nbest.JOB.gz \| \
      lattice-scale --acoustic-scale=-1 --lm-scale=-1 ark:- ark:- \| \
      lattice-compose ark:- "fstproject --project_output=true $lang_dir/G.fst |" ark:- \| \
      lattice-1best ark:- ark:- \| \
      lattice-scale --acoustic-scale=-1 --lm-scale=-1 ark:- ark:- \| \
      gzip -c \>$output_dir/nbest_nolm/nbest.JOB.gz || exit 1;

      # lattice-compose ark:- "fstproject --project_output=true $newlm |" ark:- \| \
      # lattice-determinize ark:- ark:- \| \
      # gzip -c \>$outdir/lat.JOB.gz || exit 1;


## DECOMPOSING THE N-BESTs INTO 4 ARCHIVES ##
mkdir -p $output_dir/ali/
mkdir -p $output_dir/words/
mkdir -p $output_dir/graphcost/
mkdir -p $output_dir/accost/

$cmd JOB=1:$nj $output_dir/log/decomposing.JOB.log \
  nbest-to-linear "ark:gunzip -c $output_dir/nbest_nolm/nbest.JOB.gz|" \
  "ark,t:$output_dir/ali/JOB.ali" \
  "ark,t:$output_dir/words/JOB.words" \
  "ark,t:$output_dir/graphcost/JOB.graphcost" "ark,t:$output_dir/accost/JOB.accost"

## TRANSLATING WORD-ID-BASED HYPOTHESIS TO ACTUAL WORDS ##
mkdir -p $output_dir/hyp/
for n in `seq 1 ${nj}`; do
  int2sym.pl -f 2- $lang_dir/words.txt $output_dir/words/$n.words > $output_dir/hyp/$n.hyp
done;

## RESCORING THE GRAPH-BASED SCORES USING THE NEW LM ##
mkdir -p $output_dir/newlmcost/
python3 -u rescoring.py $database $nj $output_dir/graphcost/ $output_dir/hyp/ $output_dir/newlmcost/

## RECONSTRUCTING LATTICE CONSIDERING NEW GRAPH+LM SCORES ##
mkdir -p $output_dir/rescored_nbest/
$cmd JOB=1:$nj $output_dir/log/reconstruction.JOB.log \
  linear-to-nbest "ark:$output_dir/ali/JOB.ali" \
  "ark:$output_dir/words/JOB.words" \
  "ark:$output_dir/newlmcost/JOB.newlmcost" \
  "ark:$output_dir/accost/JOB.accost" "ark:$output_dir/rescored_nbest/JOB.rnbest"

mkdir -p $output_dir/rescored_lattice/
$cmd JOB=1:$nj $output_dir/log/rnbest2lat.JOB.log \
  nbest-to-lattice "ark:$output_dir/rescored_nbest/JOB.rnbest" \
  "ark:|gzip -c >$output_dir/rescored_lattice/lat.JOB.gz" || exit 1;

## DECODING ##
scoring_opts="--min-lmwt 1 --max-lmwt 20 --beam "${beam}" --word-ins-penalty -5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0"

if [[ "$lattice_dir" == *"GMM-HMM"*  ]]
then
  ../GMM-HMM/local/score.sh --cmd "$cmd" $scoring_opts \
    $data_dir $lang_dir $output_dir/rescored_lattice/
elif [[ "$lattice_dir" == *"DNN-HMM"* ]]
then
  ../DNN-HMM/nnet1/local/score.sh --cmd "$cmd" $scoring_opts \
    $data_dir $lang_dir $output_dir/rescored_lattice/
else
  echo "Decoder not recognised!!"
fi

echo -e "\n##############################"
cat $output_dir/rescored_lattice/scoring_kaldi/best_wer
echo "##############################"
