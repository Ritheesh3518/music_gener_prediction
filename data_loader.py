import os
from feature_extractor import extract_features

def load_dataset_features(audio_path):
    genres = [g for g in os.listdir(audio_path) if os.path.isdir(os.path.join(audio_path, g))]
    features = []
    labels = []
    for genre in genres:
        genre_dir = os.path.join(audio_path, genre)
        for file in os.listdir(genre_dir):
            if file.endswith(".wav"):
                file_path = os.path.join(genre_dir, file)
                try:
                    mfcc = extract_features(file_path)
                    features.append(mfcc)
                    labels.append(genre)
                except Exception as e:
                    print(f"Could not process file {file_path}: {e}")
    return features, labels
