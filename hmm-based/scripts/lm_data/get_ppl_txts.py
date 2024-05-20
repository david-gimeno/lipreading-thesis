import os
import sys
from tqdm import tqdm

def get_ppl_txt(txt_path, lm_level, text_dst):
    text_lines = open(txt_path, "r", encoding="utf-8").readlines()
    with open(text_dst, "w") as f:
        for text_line in text_lines:
            text = " ".join(text_line.split()[1:]).strip()

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

    data_dir = sys.argv[1]
    lm_level = sys.argv[2]

    datasets = [dataset for dataset in os.listdir(data_dir) if "source-data" not in dataset]
    for dataset in datasets:
        text_path = os.path.join(data_dir, dataset, "text")
        text_dst = "./" + dataset + ".textLM.txt"
        get_ppl_txt(text_path, lm_level, text_dst)
