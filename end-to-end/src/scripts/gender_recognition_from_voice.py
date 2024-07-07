import os
import math
import random
import argparse
import pandas as pd
from tqdm import tqdm

import audeer
import audonnx
import librosa
import numpy as np

def model_building(url='https://zenodo.org/record/7761387/files/w2v2-L-robust-6-age-gender.25c844af-1.1.1.zip'):
    cache_root = audeer.mkdir('cache')
    model_root = audeer.mkdir('model')
    id2gender = {0: 'female', 1: 'male', 2: 'child'}

    archive_path = audeer.download_url(url, cache_root, verbose=True)
    audeer.extract_archive(archive_path, model_root)
    model = audonnx.load(model_root)

    return model, id2gender

def process_dataset(data_root, n_repeat=5):
    spkr_age_gender = []

    spkrs = sorted(os.listdir(data_root))[args.left:args.right]
    for spkr_id in tqdm(spkrs):
        spkr_dir = os.path.join(data_root, spkr_id)
        spkr_samples = os.listdir(spkr_dir)

        random.shuffle(spkr_samples)
        samples_to_process = spkr_samples[:n_repeat]

        spkr_ages = []
        spkr_genders = []
        for sample in samples_to_process:
            sample_path = os.path.join(spkr_dir, sample)

            if librosa.get_duration(path=sample_path) <= 20.9:
                signal, sr = librosa.load(sample_path, sr=16000)
                output_dict = model(signal, 16000)
                spkr_ages.append( output_dict['logits_age'][0][0] )
                spkr_genders.append( id2gender[np.argmax( output_dict['logits_gender'] )] )

        if len(spkr_ages) > 0:
            age = math.ceil( np.array(spkr_ages).mean() * 100 )
            gender = max(set(spkr_genders), key=spkr_genders.count)
            spkr_age_gender.append( (spkr_id, age, gender) )
        else:
            spkr_age_gender.append( (spkr_id, -1, -1) )

    return spkr_age_gender

def save_output_metadata(metadata, output_path):
    path_to_dir = os.path.dirname(output_path)
    os.makedirs(path_to_dir, exist_ok=True)

    metadata_df = pd.DataFrame(metadata, columns=['spkr_id', 'age', 'gender'])
    metadata_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    """Please refer to:
        · https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender
        · https://github.com/audeering/w2v2-age-gender-how-to?tab=readme-ov-file
    """

    # -- command line arguments
    parser = argparse.ArgumentParser(description="Automatic Age and Gender Detection from Voice.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data-root", default="../../LRS2-BBC/WAVs/", type=str, help="Directory with one directory for each speaker and her/his corresponding speech utterances")
    parser.add_argument("--n-repeat", default=5, type=int, help="Number of speaker samples on to apply the majority voting decidision taking")
    parser.add_argument("--left", default=0, type=int, help="Left margin to split speaker and then parallelize extraction")
    parser.add_argument("--right", default=7295, type=int, help="Right margin to split speaker and then parallelize extraction")
    parser.add_argument("--output-path", required=True, type=str, help="Path where the resulting CSV will be stored.")

    args = parser.parse_args()


    model, id2gender = model_building()
    metadata = process_dataset(args.data_root, args.n_repeat)
    save_output_metadata(metadata, args.output_path)
