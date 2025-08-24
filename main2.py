import os
import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings('ignore')

# 1. Define Objective
print("Objective: Classify music tracks into genres using ML.")

# 2. Load Dataset - read genres from subdirectories
audio_path = "/Users/admin/Desktop/GIT/Riteesh/music/Data/genres_original"
genres = [g for g in os.listdir(audio_path) if os.path.isdir(os.path.join(audio_path, g))]
print("Genres found:", genres)

# Function to load audio using pydub (works around librosa backend issues)
def load_audio_with_pydub(file_path, sr=22050, duration=30):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(sr).set_channels(1)
    audio_samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    audio_samples /= np.max(np.abs(audio_samples))  # Normalize
    # Trim or pad to desired duration
    expected_len = sr * duration
    if len(audio_samples) > expected_len:
        audio_samples = audio_samples[:expected_len]
    else:
        audio_samples = np.pad(audio_samples, (0, max(0, expected_len - len(audio_samples))))
    return audio_samples, sr

# Updated feature extraction to use pydub loader
def extract_features(file_path, sr=22050, duration=30):
    y, sr = load_audio_with_pydub(file_path, sr, duration)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# 4 & 5. Preprocess & Feature Extraction for all audio
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
                
features = np.array(features)
labels = np.array(labels)
print(f"Extracted features from {len(features)} audio files.")

# 6. Exploratory Data Analysis (PCA visualization)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(features_scaled)

plt.figure(figsize=(10,8))
sns.scatterplot(x=principal_components[:,0], y=principal_components[:,1], hue=labels, palette='Set2', legend='full')
plt.title('PCA of MFCC Features from GTZAN')
plt.show()

# 7. Prepare Dataset for ML
le = LabelEncoder()
y_encoded = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(features_scaled, y_encoded, test_size=0.2,
                                                    random_state=42, stratify=y_encoded)

# 8. Train Traditional ML Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=500),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Gradient Boosting': GradientBoostingClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} trained.")

# 9. Evaluate ML Models
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()

# 10. Tune Hyperparameters (example for Random Forest)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
best_rf = grid.best_estimator_

# Evaluate tuned model
y_pred = best_rf.predict(X_test)
print("\nTuned Random Forest Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 15. Save & Export Model
joblib.dump(best_rf, "random_forest_genre_classifier.joblib")
print("Random Forest model saved.")

joblib.dump(scaler, "feature_scaler.joblib")
print("Scaler saved.")

joblib.dump(le, "label_encoder.joblib")
print("Label encoder saved.")
