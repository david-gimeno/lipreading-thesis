import os
import sys
import math
import random
import pandas as pd
from tqdm import tqdm

"""
  LRS2-BBC          ||  LRS3-TED          ||  LIP-RTVE
                    ||                    ||
  0.5% ~ 1 hour     ||  0.32% ~ 1 hour    ||  11.0% ~ 1 hour
  1.0% ~ 2 hours    ||  0.64% ~ 2 hours   ||  22.0% ~ 2 hours
  2.5% ~ 5 hours    ||  1.60% ~ 5 hours   ||  54.0% ~ 5 hours
  5.0% ~ 10 hours   ||  3.20% ~ 10 hours  ||
  10.0% ~ 20 hours  ||  6.40% ~ 20 hours  ||
  24.0% ~ 50 hours  ||  16.0% ~ 50 hours  ||
  48.0% ~ 100 hours ||  32.0% ~ 100 hours ||
                    ||  64.0% ~ 200 hours ||
"""

## CONFIGURATION ##
database = sys.argv[1]
scenario = sys.argv[2]
prctg = float(sys.argv[3])
previous = sys.argv[4]
dst_path = sys.argv[5]

dataset = "fulltrain" if "LRS" in database else "train"
fps = 50 if "VLRF" == database else 25

## READING DATAFRAME CONTAINING PARTITION DATASET ##
fulltrain = pd.read_csv(os.path.join("../data/", database, "splits", scenario, dataset+database+".csv"))
if previous != "none":
    previous_data = pd.read_csv(os.path.join("../data/", database, "splits", scenario, previous+database+".csv"))["sampleID"].tolist()
    print(dataset+database+".csv was read:", len(fulltrain), "samples.")

    dataframe = fulltrain[~fulltrain["sampleID"].isin(previous_data)]
    print(dataset+database+".csv was read:", len(dataframe), "samples.")
    print("Removing sampleIDs from:", previous+database+".csv (" + str(len(previous_data)), "frames)")
    print("Considering therefore", len(dataframe), "samples")
else:
    dataframe = fulltrain
    print(dataset+database+".csv was read:", len(dataframe), "samples.")

## SPLITTING DATAFRAME DEPENDING ON THE NUMBER OF FRAMES THAT COMPOSED EACH SAMPLE ##
frame_splits = {}
for min_frames, max_frames in [(0, 100), (100,150), (150,300), (300,450), (450,600)]:
    frame_data = dataframe[(dataframe["nFrames"] > min_frames) & (dataframe["nFrames"] <= max_frames)]
    frame_splits[str(max_frames)+"frames"] = frame_data["sampleID"].tolist()

## RANDOM SELECTION OF EACH FRAME SPLIT BASED ON THE SPECIFIED PERCENTAGE ##
new_dataset = []
random.seed(12)
for k in frame_splits.keys():
    frame_split = frame_splits[k]
    length = len(frame_split)
    new_length = math.ceil((prctg * length) / 100)

    sample_prctg = random.sample(frame_split, new_length)
    new_dataset += sample_prctg

if previous != "none":
    new_dataset += previous_data

## COMPUTING TIME IN TERMS OF SECONDS ##
# total_frames = 0.0
# new_data = []
# for sampleID in tqdm(fulltrain["sampleID"].tolist()):
#     if sampleID in new_dataset:
#         nframes = fulltrain[fulltrain["sampleID"]==sampleID]["nFrames"].values[0]
#         total_frames += nframes
#         new_data.append( (sampleID, nframes) )
# total_seconds = total_frames / fps

new_dataframe = fulltrain[fulltrain["sampleID"].isin(new_dataset)]
total_seconds = new_dataframe["nFrames"].sum() / fps
total_hours = ((total_seconds/60)/60)
## LOGGING INFO ##
print("The new dataset is composed of", len(new_dataframe), "samples and it offers of around", total_seconds, "seconds of data (", total_hours, "hours ).")
dst_dir = "../data/" + database + "/splits/" + scenario + "/"
# new_dataframe = pd.DataFrame(new_data, columns=["sampleID", "nFrames"])
print("Writing CSV in", dst_dir+"/"+dst_path)
new_dataframe.to_csv(dst_dir+"/"+dst_path)
