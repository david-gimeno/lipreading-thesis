import os
import sys
import numpy as np
from tqdm import tqdm
from scipy.interpolate import CubicSpline
from kaldiio import WriteHelper, ReadHelper

def alignVisualFeats(features, targetFPS):
   f, e = features.shape
   newFeats = []

   for k in range(0, e):
       x = np.array(range(f)) # Depedenra del numero de frames del sample
       y = features[:, k] #Seleccinamos la columna de la matrix para el componente k-esimo
       #xi = np.arange(0, f, targetFPS) # Interpolamos el numero de frames nuevo
       xi = np.linspace(0, f, targetFPS)

       #print("x", x.shape, type(x))
       #print("y", y.shape, type(y))
       #print("xi", xi.shape, type(xi))

       inter = CubicSpline(x, y)
       yy = inter(xi)
       #print("yy", yy.shape)
       newFeats.append(yy)

   newFeats = np.array(newFeats).T
   #print(newFeats.shape)

   return newFeats

def extractFeatures(root_source, root_dst, numFeats, fps):
    print("\n" + "-"*60)
    print("Preparing the Visual Speech Features")

    datasets = sorted(os.listdir(root_source))

    for dataset in datasets:
        totalVectors = 0
        pathARK = os.getcwd() + "/" + root_source + dataset
        pathSCP = os.getcwd() + "/" + root_dst + dataset

        if not os.path.exists(pathSCP+"/feats.scp"):
            print("Computing features for", dataset, "dataset")
            seconds = 0
            with WriteHelper("ark,scp:" + pathARK + "/feats.ark," + pathSCP + "/feats.scp") as writer:
                path2dataset = root_source + dataset + "/"

                for spkr in tqdm(sorted(os.listdir(path2dataset))):
                    if spkr not in ["feats.ark", "feats.scp"]:
                        for sample in sorted(os.listdir(path2dataset+spkr+"/")):
                            sampleID = sample.split(".")[0]
                            spkrID = sampleID.split("_")[0]

                            path2features = root_source + dataset + "/" + spkrID + "/" + sample

                            if sample.split(".")[-1] == "npz":
                                sequence = np.load(path2features)
                                features = sequence[sequence.files[0]]
                                nrows = features.shape[0]
                            else:
                                sequence = open(path2features, "r").readlines()[0]
                                featuresUtterance = np.fromstring(sequence, dtype=np.float32, sep=",")
                                features = featuresUtterance.reshape( (nrows, numFeats) )
                                nrows = featuresUtterance.shape[0] // numFeats

                            totalVectors += nrows
                            seconds += nrows / fps

                            #align_features = alignVisualFeats(features, mfccs[sampleID])
                            writer(sampleID, features)
                            #writer(sampleID, align_features)
                            #writer[sampleID] = featuresUtterance

            print("\nSeconds in " + dataset + " set: ", round(seconds,2))
            print("#FeatureVectors: ", totalVectors)

if __name__ == "__main__":
    print("#################################")
    print("VISUAL SPEECH FEATURES EXTRACTION")
    print("#################################\n")

    if len(sys.argv) < 2:
        print("Error in invokation: Usage example: python3 local/getFeats.py 32")
        sys.exit(0)

    root_source = str(sys.argv[1])
    root_dst = str(sys.argv[2])
    numFeats = int(sys.argv[3])
    fps = int(sys.argv[4].split("fps")[0])

    ## ADJUSTING VISUAL STREAM SAMPLE RATE WITH ACOUSTIC SIGNAL  ##
    # mfccs={}
    # if os.path.exists("./data/audio100fps/train/feats.scp"):
    #     for dataset in ["train", "dev", "test"]:
    #         with ReadHelper("scp:./data/audio100fps/" + dataset + "/feats.scp") as reader:
    #             for key, feats in reader:
    #                 print(key, feats.shape[0])
    #                 mfccs[key] = feats.shape[0]

    extractFeatures(root_source, root_dst, numFeats, fps) #, mfccs)

    print("\n###############################")
    print("VISUAL SPEECH FEATURES COMPLETED")
    print("#################################\n")
