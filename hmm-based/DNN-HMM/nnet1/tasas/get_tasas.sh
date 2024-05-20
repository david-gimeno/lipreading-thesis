#!/bin/bash

decode_dir=$1

sort ${decode_dir}/scoring_kaldi/test_filt.txt | cut -f 2- -d ' ' > ./tasas/groundtruth.txt

output=$(cat ${decode_dir}/scoring_kaldi/best_wer | rev | cut -d/ -f1 | rev)
lmwt=$(echo $output | cut -f2 -d_)
penalty=$(echo $output | cut -f3 -d_)

sort ${decode_dir}/scoring_kaldi/penalty_${penalty}/${lmwt}.txt | cut -f 2- -d ' ' > ./tasas/pred_${lmwt}_${penalty}.txt
paste -d '#' ./tasas/groundtruth.txt ./tasas/pred_${lmwt}_${penalty}.txt > ./tasas/hyp_${lmwt}_${penalty}.txt

echo -e "\n%WER of"${decode_dir}"\n"
./tasas/tasas -s " " -f "#" -ie ./tasas/hyp_${lmwt}_${penalty}.txt
./tasas/tasasIntervalo -s " " -f "#" -ie ./tasas/hyp_${lmwt}_${penalty}.txt
echo ""

# if [[ "$system" == *"smbr"* || "$system" == *"mmi"* || "$system" == *"mpe"* ]]; then
#    best_iter=1
#    best_wer=500.0
#    for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
#         out=$(cat ../exp/${modality}/${system}/decode_test_${i}/scoring_kaldi/best_wer)
#         wer=$(echo $out | cut -f 2 -d ' ')
#         if [[ $(echo "$wer < $best_wer" | bc -l) == 1 ]]; then best_iter=$i; best_wer=$wer; fi;

#    done;
#    output=$(cat ../exp/${modality}/${system}/decode_test_${best_iter}/scoring_kaldi/best_wer)
#    lmwt=$(echo $output | cut -f5 -d_)
#    penalty=$(echo $output | cut -f6 -d_)
#    sort ../exp/${modality}/${system}/decode_test_${best_iter}/scoring_kaldi/penalty_${penalty}/${lmwt}.txt | cut -f 2- -d ' ' > ./pred_${lmwt}_${penalty}.txt
# fi

rm ./tasas/groundtruth.txt
rm ./tasas/hyp_*
rm ./tasas/pred_*
