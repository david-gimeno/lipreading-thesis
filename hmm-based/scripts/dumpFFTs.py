import os
import sys
import numpy as np
import pickle as pkl
from kaldiio import ReadHelper
from tqdm import tqdm

input_dir = sys.argv[1]
output_dir = sys.argv[2]
for partition in tqdm(["train", "dev", "test"]):
    with ReadHelper("scp:"+input_dir+partition+"/feats.scp") as reader:
        for sampleID, effts in reader:
            ffts = effts[:, 1:]
            #print(sampleID, ffts.shape, type(ffts), ffts[0].shape, type(ffts[0]))
            spkrID = sampleID.split("_")[0]
            os.makedirs(os.path.join(output_dir, spkrID), exist_ok=True)

            #for i, featureVector in enumerate(ffts):
            #    dstPath = os.path.join(output_dir, spkrID, sampleID, sampleID + "_" + str(i).zfill(4) + ".npz")
            dstPath = os.path.join(output_dir, spkrID, sampleID+".npz")
            np.savez_compressed(dstPath, data=ffts)

