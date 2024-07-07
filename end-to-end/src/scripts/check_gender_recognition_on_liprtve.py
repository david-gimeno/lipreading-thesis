import os
import math
import random
import argparse
import pandas as pd

import audeer
import audonnx
import librosa
import numpy as np

if __name__ == "__main__":
    """Please refer to:
        · https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender
        · https://github.com/audeering/w2v2-age-gender-how-to?tab=readme-ov-file
    """
    # -- command line arguments
    data_root = '../../data/LIP-RTVE/WAVs/'
    reference_path = './liprtve_gender_metadata.csv'
    reference = pd.read_csv(reference_path)

    id2gender = {0: 'female', 1: 'male', 2: 'child'}

    url = 'https://zenodo.org/record/7761387/files/w2v2-L-robust-6-age-gender.25c844af-1.1.1.zip'
    cache_root = audeer.mkdir('cache')
    model_root = audeer.mkdir('model')

    archive_path = audeer.download_url(url, cache_root, verbose=True)
    audeer.extract_archive(archive_path, model_root)
    model = audonnx.load(model_root)


    ncorrect = 0
    nincorrect = 0
    for spkr_id in reference['speaker_id'].tolist():
        spkr_dir = os.path.join(data_root, spkr_id)
        spkr_samples = os.listdir(spkr_dir)

        random.shuffle(spkr_samples)
        samples_to_process = spkr_samples[:5]

        ages = []
        genders = []
        for sample in samples_to_process:
            sample_path = os.path.join(spkr_dir, sample)
            signal, sr = librosa.load(sample_path, sr=16000)

            output_dict = model(signal, 16000)

            age = output_dict['logits_age'][0][0]; ages.append(age)
            gender = id2gender[np.argmax( output_dict['logits_gender'] )]; genders.append(gender)

        age = math.ceil( np.array(ages).mean() * 100 )
        gender = max(set(genders), key=genders.count)

        ref_gender = reference[reference['speaker_id'] == spkr_id]['gender_id'].values[0]
        if gender == ref_gender:
            ncorrect += 1
        else:
            nincorrect += 1
            print(f'There is a mismatch with speaker {spkr_id}, the model predicts "{gender}", while you said "{ref_gender}"')
        print(f'Correct: {ncorrect} | Incorrect: {nincorrect}', end='\r')
        # print(spkr_id, ": ", gender, " -- ", ref_gender)
