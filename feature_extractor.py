from pydub import AudioSegment
import numpy as np
import librosa

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
