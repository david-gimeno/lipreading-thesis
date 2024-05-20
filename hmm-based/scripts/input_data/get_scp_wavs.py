import os
import sys

root_source = str(sys.argv[1])
root_dst = str(sys.argv[2])

datasets = sorted(os.listdir(root_source))

for dataset in datasets:
    dst_path = root_dst + dataset + "/wav.scp"
    if not os.path.exists(dst_path):
        writer = open(dst_path, "a", encoding="utf-8")

        path2dataset = root_source + dataset + "/"
        spkrs = sorted(os.listdir(path2dataset))
        for spkr in spkrs:
            path2spkr = path2dataset + spkr + "/"

            wavs = sorted(os.listdir(path2spkr))
            for wav in wavs:
                sampleID = wav.split(".")[0]
                path2wav = path2spkr + wav
                writer.write(sampleID + " " + path2wav + "\n")

        writer.close()
