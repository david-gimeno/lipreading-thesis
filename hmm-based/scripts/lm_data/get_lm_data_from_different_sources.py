import os
import sys
import pandas as pd
from tqdm import tqdm

sys.path.insert(0,'./scripts/')
from my_utils import *

def get_lm_data(text_dir, sampleIDs, lm_level):
    text_samples = []
    for sampleID in tqdm(sampleIDs):
        spkrID = sampleID[:-6]
        sample_path = os.path.join(text_dir, spkrID, sampleID+".txt")

        ## CLEANING TEXT DATA ACCORDING TO THE LANGUAGE ##
        text = open(sample_path, "r", encoding="utf-8").readlines()[0].strip()
        text = clean_english_text(text).strip()

        ## DEPENDING ON THE SPECIFIED LEVEL ##
        if "character" in lm_level:
            text = " ".join(list(map(lambda x: x.replace(" ", "<space>"), text)))

        text_samples.append(text)

    return text_samples

if __name__ == "__main__":
    """
        Extracting data from the training set, detailed in a CSV file.

    lm_level: indicates if the LM is based on characters or words.
    split_path: path where the CSV that defines the training set is
    lmdata_dir: root dir where the resulting TXT file will be store
    """

    lm_level = "word"
    dst_path = "./specificLM/lrs2+lrs3.txt"
    splits = ["../data/LRS2-BBC/splits/speaker-independent/fulltrainLRS2-BBC.csv", "../data/LRS3-TED/splits/speaker-independent/fulltrainLRS3-TED.csv"]

    all_text_samples = []
    for split_path in splits:
        text_dir = "/".join(split_path.split("/")[:3]) + "/transcriptions/"

        sampleIDs = pd.read_csv(split_path)["sampleID"].tolist()
        all_text_samples += get_lm_data(text_dir, sampleIDs, lm_level)

    with open(dst_path, "w", encoding="utf-8") as f:
        for text_sample in all_text_samples:
            f.write(text_sample.strip() + "\n")

