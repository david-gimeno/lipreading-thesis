import os
import sys
import numpy as np
import pickle as pkl
from kaldiio import ReadHelper
from tqdm import tqdm

input_dir = sys.argv[1]
output_dir = sys.argv[2]

datasets = [d for d in sorted(os.listdir(input_dir)) if d in ["train", "dev", "test"]]
for partition in tqdm(datasets):
    with ReadHelper("scp:"+input_dir+partition+"/feats.scp") as reader:
        for sampleID, mfccs in reader:
            #print(sampleID, mfccs.shape, type(mfccs), mfccs[0].shape, type(mfccs[0]))
            spkrID = sampleID.split("_")[0]
            os.makedirs(os.path.join(output_dir, spkrID), exist_ok=True)

            dstPath = os.path.join(output_dir, spkrID, sampleID+".npz")
            np.savez_compressed(dstPath, data=mfccs)

