import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


DATA = "./data/external/"
df = pd.read_csv(DATA + "spotify.csv")

features = ['popularity', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'mode', 
            'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
X = df[features]
y = df['track_genre']

def normalize(X):
    return (X - X.min()) / (X.max() - X.min())

X_normalized = normalize(X)

genre_mapping = {genre: i for i, genre in enumerate(y.unique())}
y_encoded = y.map(genre_mapping)

def train_val_test_split(X, y, val_size=0.15, test_size=0.15, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    val_size = int(len(X) * val_size)
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size+val_size]
    train_indices = indices[test_size+val_size:]
    return (X.iloc[train_indices], X.iloc[val_indices], X.iloc[test_indices],
            y.iloc[train_indices], y.iloc[val_indices], y.iloc[test_indices])

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X_normalized, y_encoded)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
save_path = 'data/interim/a2/'
os.makedirs(save_path, exist_ok=True)
np.save(save_path + 'X_train.npy', X_train)
np.save(save_path + 'X_val.npy', X_val)
np.save(save_path + 'X_test.npy', X_test)
np.save(save_path + 'y_train.npy', y_train)
np.save(save_path + 'y_val.npy', y_val)
np.save(save_path + 'y_test.npy', y_test)