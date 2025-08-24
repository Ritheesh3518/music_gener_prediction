# Music Genre Classification

This project classifies music tracks into genres using machine learning. It uses MFCC features extracted from `.wav` files and supports a variety of classifiers. The pipeline includes feature extraction, preprocessing, model training/tuning, evaluation, and model persistence.

## Folder Structure

```
music_project/
│
├── data_loader.py             # Dataset loading and feature extraction
├── feature_extractor.py       # Audio loading and feature extraction functions
├── data_preprocessing.py      # Scaling, encoding, splitting functions
├── model_training.py          # ML model training and evaluation
├── hyperparameter_tuning.py   # Hyperparameter tuning functions
├── model_persistence.py       # Save/load model and scalers
└── main.py                    # Main script tying everything together
```

## Dataset

- Expects audio organized by genre:
  ```
  genres_original/
    jazz/
      jazz.00000.wav
      jazz.00001.wav
      ...
    rock/
      ...
    (etc.)
  ```
- Update the `audio_path` in `main.py` to your local dataset path if different.

## Requirements

- Python 3.8+
- numpy
- librosa
- pydub
- scikit-learn
- joblib
- matplotlib
- seaborn

**Additional Dependency (for PyDub):**
- FFmpeg (must be installed and accessible in your system’s PATH).

Install all Python dependencies:

```bash
pip install -r requirements.txt
```

#### Install FFmpeg

- On macOS with Homebrew:

  ```bash
  brew install ffmpeg
  ```

- On Ubuntu:

  ```bash
  sudo apt-get install ffmpeg
  ```

- [Other platforms and troubleshooting](https://stackoverflow.com/questions/tagged/pydub):  
  Make sure `ffmpeg` is in your PATH so that PyDub can find it automatically.
  If you experience input errors or "ffprobe not found" errors, ensure the file is a valid `.wav` file and not corrupted.[1][3][4]

## Usage

1. Prepare your dataset as described in “Dataset”.
2. Edit the `audio_path` in `main.py` if necessary.
3. Run:
   ```bash
   python main.py
   ```
4. You’ll see classification reports and confusion matrices for each model.
5. The tuned best model, scaler, and label encoder will be saved as `.joblib` files.

## Results (Sample Output)

From a recent run:
- **Random Forest (tuned):**  
  Accuracy: 0.62  
  Macro F1: 0.61 (see the script output for detailed per-genre scores).  
- **Logistic Regression, SVM, KNN, Gradient Boosting:**  
  Accuracies from 0.50 to 0.59

Some genres, like `classical` and `metal`, consistently showed higher recall and precision, while genres like `disco` and `rock` were harder to classify.

## Troubleshooting

- **FFmpeg input errors:**  
  Make sure files are not corrupt and are valid `.wav` format. Invalid files will be skipped and a message printed.
- **FFmpeg not found:**  
  Add ffmpeg to your system PATH, or specify the path to it in your code using:
  ```python
  from pydub.utils import which
  AudioSegment.converter = which("ffmpeg")
  ```
  [See PyDub/FFmpeg issues for more.][4][5]

- **Corrupted/Invalid file warning:**  
  The pipeline skips invalid or unreadable files and processes the rest.

## Customization & Extensions

- Add more audio features in `feature_extractor.py`
- Plug in deep learning models if needed (e.g., CNNs on spectrograms)
- Adjust hyperparameters in `hyperparameter_tuning.py`