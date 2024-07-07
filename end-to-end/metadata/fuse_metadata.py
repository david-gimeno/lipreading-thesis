import sys
import pandas as pd

if __name__ == "__main__":
    dataset_path = sys.argv[1]
    metadata_path = sys.argv[2]
    output_path = sys.argv[3]

    dataset = pd.read_csv(dataset_path, index_col=0)
    metadata = pd.read_csv(metadata_path)

    dataset['speakerID'] = dataset['sampleID'].map(lambda x: x[:-6])
    dataset['gender'] = dataset['speakerID'].map(lambda x: metadata[metadata['spkr_id'] == x]['gender'].values[0])
    dataset['age'] = dataset['speakerID'].map(lambda x: metadata[metadata['spkr_id'] == x]['age'].values[0])

    dataset.drop('speakerID', axis=1, inplace=True)

    dataset.to_csv(output_path)
