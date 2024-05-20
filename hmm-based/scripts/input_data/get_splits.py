import os
import sys
import random
import shutil
import pandas as pd
from tqdm import tqdm

def readPartition(database, scenario):
    rootSplit = "../data/" + database + "/splits/" + "/" + scenario + "/"
    # LRS3-TED
    #fullTrainList = pd.read_csv(rootSplit + "train" + database  + ".csv", delimiter=",")
    #trainList = fullTrainList[fullTrainList["nFrames"] <= 100]["uttID"].tolist() #[:12500] # ~30 hours of data; 20 seconds maximum length
    #devList = pd.read_csv(rootSplit + "dev" + database  + ".csv", delimiter=",")["sampleID"].tolist()[:1250] # ~1 h

    if database == "LRS3-TED":
        # splits = ["pretrain", "trainval", "test"]
        splits = ["1hour", "2hours", "5hours", "10hours", "20hours", "50hours", "100hours", "200hours", "fulltrain", "test"]
    elif database == "LRS2-BBC":
        # splits = ["fulltrain"]
        # splits = ["fulltrain", "train", "dev", "test"]
        splits = ["1hour", "2hours", "5hours", "10hours", "20hours", "50hours", "100hours", "fulltrain", "dev", "test"]
    elif database == "VLRF":
        splits = ["fulltrain", "test"]
    elif database == "LIP-RTVE":
        splits = ["train", "dev", "test"]

    datasets = []
    for split in splits:
        all_data = pd.read_csv(rootSplit + split + database + ".csv", delimiter=",") # ["sampleID"].tolist()
        data = all_data[all_data["nFrames"]<=600]
        samples = data["sampleID"].tolist()
        datasets.append( (split, samples) )

    totalsamples = 0
    for name, dataset in datasets:
        totalsamples += len(dataset)
        print("\#"+name+" SAMPLES:", len(dataset))
    print("\#DATABASE SAMPLES:", totalsamples,"\n")

    return datasets

def getPart(modality, root_data, database, scenario, dst_path):
    datasets = readPartition(database, scenario)
    extension = ".wav" if "audio" in modality else ".npz"

    ## DATA MUST BE STRUCTURED AS KALDI DEMANDS ##
    for name, dataset in datasets:
        for sample in tqdm(dataset):
            spkrID = sample.split("_")[0]
            root_dst = dst_path + name.lower() + "/" + spkrID + "/"
            os.makedirs(root_dst, exist_ok=True)

            src_sample = root_data + "/" + spkrID + "/" + sample + extension
            dst_sample = root_dst + "/" + sample + extension
            shutil.copy(src_sample, dst_sample)

    print("\n############################")
    print("DATABASE PARTITION COMPLETED")
    print("############################")

if __name__ == "__main__":
    root_data = str(sys.argv[1])
    database = str(sys.argv[2])
    modality = str(sys.argv[3])
    dst_path = str(sys.argv[4])
    scenario = str(sys.argv[5])

    os.makedirs(dst_path, exist_ok=True)

    getPart(modality, root_data, database, scenario, dst_path)
