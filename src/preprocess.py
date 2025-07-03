import librosa
import numpy as np
import os

def extract_features(file_path, sr=22050, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

def preprocess_data(directory):
    features = []
    labels = []
    for label in os.listdir(directory):
        class_dir = os.path.join(directory, label)
        if not os.path.isdir(class_dir):
            continue
        for file in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file)
            features.append(extract_features(file_path))
            labels.append(label)
    return np.array(features), np.array(labels)
