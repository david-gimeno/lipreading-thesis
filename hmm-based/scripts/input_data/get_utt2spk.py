import os
import sys
import random
import shutil

#$rootsource $database $modality $scenario
root_source = str(sys.argv[1])
root_dst = str(sys.argv[2])
database = str(sys.argv[3])

datasets = sorted(os.listdir(root_source))

for dataset in datasets:
    dst_path = root_dst + dataset + "/utt2spk"
    if not os.path.exists(dst_path):
        writer = open(dst_path, 'a')

        path2dataset = root_source + dataset + "/"
        spkrs = sorted(os.listdir(path2dataset))
        for spkrID in spkrs:
            path2spkr = path2dataset + '/' + spkrID + '/'
            samples = sorted(os.listdir(path2spkr))

            for sample in samples:
                sampleID = sample.split('.')[0]
                writer.write(sampleID + ' ' + spkrID + '\n')

        writer.close()

print('\n##################################')
print("FILES UTT2SPK and SPK2UTT COMPLETED")
print('###################################\n')
