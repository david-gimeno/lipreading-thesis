import os
import sys
import pandas as pd
from tqdm import tqdm

sys.path.insert(0,'./scripts/')
from my_utils import *

def get_lm_data(text_dir, sampleIDs, lm_level, lmdata_dst):
    database = text_dir.split("/")[2]

    delimiter = 6 if database in ["LRS3-TED", "LRS2-BBC"] else 5
    enc = "ISO-8859-1" if database == "VLRF" else "utf-8"

    with open(lmdata_dst, "w", encoding=enc) as f:
        for sampleID in tqdm(sampleIDs):
            spkrID = sampleID[:-delimiter]
            sample_path = os.path.join(text_dir, spkrID, sampleID+".txt")

            ## CLEANING TEXT DATA ACCORDING TO THE LANGUAGE ##
            text = open(sample_path, "r", encoding=enc).readlines()[0].strip()
            if database in ["LRS2-BBC", "LRS3-TED"]:
                text = clean_english_text(text).strip()
            elif database in ["CMU-MOSEAS-Spanish", "VLRF", "LIP-RTVE"]:
                text = clean_spanish_text(text).strip()

            ## DEPENDING ON THE SPECIFIED LEVEL ##
            if "character" in lm_level:
                text = " ".join(list(map(lambda x: x.replace(" ", "<space>"), text)))

            f.write(text + "\n")

if __name__ == "__main__":
    """
        Extracting data from the training set, detailed in a CSV file.

    lm_level: indicates if the LM is based on characters or words.
    split_path: path where the CSV that defines the training set is
    lmdata_dir: root dir where the resulting TXT file will be store
    """

    lm_level = sys.argv[1]
    split_path = sys.argv[2]
    lmdata_dir = sys.argv[3]

    database = split_path.split("/")[2]
    delimiter = 6 if database in ["LRS3-TED", "LRS2-BBC"] else 5

    text_dir = "/".join(split_path.split("/")[:3]) + "/transcriptions/"
    sampleIDs = pd.read_csv(split_path)["sampleID"].tolist()
    lmdata_dst = os.path.join(lmdata_dir, "train_lm.txt")

    get_lm_data(text_dir, sampleIDs, lm_level, lmdata_dst)
