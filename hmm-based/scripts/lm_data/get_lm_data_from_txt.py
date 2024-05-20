import os
import re
import sys
import pandas as pd
from tqdm import tqdm

sys.path.insert(0,'./scripts/')
from my_utils import *

def get_lm_data(txt_path, lm_level, lmdata_dst):
    database = lmdata_dst.split("/")[2]
    text_lines = open(txt_path, "r", encoding="utf-8").readlines()

    with open(lmdata_dst, "w") as f:
        for text_line in tqdm(text_lines):
            ## CLEANING TEXT DATA ACCORDING TO THE LANGUAGE ##
            text = text_line.strip()
            if database in ["LRS2-BBC", "LRS3-TED"]:
                text = clean_english_text(text).strip()
            elif database in ["CMU-MOSEAS-Spanish", "VLRF", "LIP-RTVE"]:
                text = clean_spanish_text(text).strip()

            for special_char in ';:,.!"?()+-_{}=#%&$[]^/\\':
                text = text.replace(special_char, "")
            text = text.replace(" ' ", "").replace("\'\'", "")
            text = re.sub(r"^'", "", text)
            text = re.sub(r"'$", "", text)

            ## DEPENDING ON THE SPECIFIED LEVEL ##
            if "character" in lm_level:
                text = " ".join(list(map(lambda x: x.replace(" ", "<space>"), text)))

            f.write(text + "\n")

if __name__ == "__main__":
    """
        Extracting data from a specified TXT file

    lm_level: indicates if the LM is based on characters or words.
    txt_path: path where the TXT that defines the data for training the LM is
    lmdata_dir: root dir where the resulting TXT file will be store
    """

    lm_level = sys.argv[1]
    txt_path = sys.argv[2]
    lmdata_dir = sys.argv[3]

    lmdata_dst = os.path.join(lmdata_dir, "train_lm.txt")
    get_lm_data(txt_path, lm_level, lmdata_dst)
