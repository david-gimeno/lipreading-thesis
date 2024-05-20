import os
import sys
from unidecode import unidecode
from num2words import num2words

sys.path.insert(0,'./scripts/')
from my_utils import *

if __name__ == "__main__":
    root_source = str(sys.argv[1])
    root_dst = str(sys.argv[2])
    database = str(sys.argv[3])

    datasets = sorted(os.listdir(root_source))

    enc = "ISO-8859-1" if database == "VLRF" else "utf-8"
    delimiter = 6 if database in ["LRS3-TED", "LRS2-BBC"] else 5

    for dataset in datasets:
        os.makedirs(root_dst + dataset, exist_ok=True)
        dst_path = root_dst + dataset + "/text"
        if not os.path.exists(dst_path):
            writer = open(dst_path, "a", encoding=enc)

            path2dataset = root_source + dataset + "/"
            spkrs = sorted(os.listdir(path2dataset))
            for spkr in spkrs:
                path2spkr = path2dataset + spkr + "/"

                samples = sorted(os.listdir(path2spkr))
                for sample in samples:
                    sampleID = sample.split(".")[0]
                    spkrID = sample.split(".")[0][:-delimiter]

                    transcription_path = "../data/" + database + "/transcriptions/" + spkrID + "/" + sampleID + ".txt"
                    transcription = open(transcription_path, "r", encoding=enc).readlines()[0].strip()
                    if database in ["LIP-RTVE", "VLRF", "CMU-MOSEAS"]:
                        transcription = clean_spanish_text(transcription)
                    elif database in ["LRS3-TED", "LRS2-BBC"]:
                        transcription = clean_english_text(transcription)
                    writer.write(sampleID + " " + transcription + "\n")

            writer.close()

print("\n####################")
print("TEXT FILES COMPLETED")
print("#####################")
