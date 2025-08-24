import streamlit as st
import numpy as np
import librosa
from pydub import AudioSegment
import joblib

# Load saved model and objects
model = joblib.load("random_forest_genre_classifier.joblib")
scaler = joblib.load("feature_scaler.joblib")
le = joblib.load("label_encoder.joblib")

# Feature extraction functions (same as before)
def load_audio_with_pydub(file_path, sr=22050, duration=30):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(sr).set_channels(1)
    audio_samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    audio_samples /= np.max(np.abs(audio_samples))
    expected_len = sr * duration
    if len(audio_samples) > expected_len:
        audio_samples = audio_samples[:expected_len]
    else:
        audio_samples = np.pad(audio_samples, (0, max(0, expected_len - len(audio_samples))))
    return audio_samples, sr

def extract_features(file_path, sr=22050, duration=30):
    y, sr = load_audio_with_pydub(file_path, sr, duration)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Streamlit UI
st.title("Music Genre Classifier 🎵")
st.write("Upload a WAV audio file (up to 30 seconds) to predict its music genre.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "flac"])

if uploaded_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio("temp_audio.wav", format='audio/wav')
    st.write("Extracting features and making prediction...")

    try:
        features = extract_features("temp_audio.wav")
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)
        predicted_genre = le.inverse_transform(prediction)[0]
        st.success(f"Predicted Genre: {predicted_genre}")
    except Exception as e:
        st.error(f"Could not process the audio file: {e}")
