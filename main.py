from data_loader import load_dataset_features
from data_preprocessing import preprocess_features_labels
from model_training import get_models, train_and_evaluate_models
from hyperparameter_tuning import tune_random_forest
from model_persistence import save_model_objects
import matplotlib.pyplot as plt

def main():
    print("Objective: Classify music tracks into genres using ML.")

    audio_path = "/Users/admin/Desktop/GIT/Riteesh/music/Data/genres_original"
    features, labels = load_dataset_features(audio_path)
    print(f"Extracted features from {len(features)} audio files.")

    X_train, X_test, y_train, y_test, scaler, le = preprocess_features_labels(features, labels)

    # Optional PCA visualization if needed
    # import seaborn as sns
    # from sklearn.decomposition import PCA
    # features_scaled = scaler.transform(features)
    # pca = PCA(n_components=2)
    # principal_components = pca.fit_transform(features_scaled)
    # plt.figure(figsize=(10,8))
    # sns.scatterplot(x=principal_components[:,0], y=principal_components[:,1], hue=labels, palette='Set2', legend='full')
    # plt.title('PCA of MFCC Features from GTZAN')
    # plt.show()

    models = get_models()
    train_and_evaluate_models(models, X_train, X_test, y_train, y_test, le)

    best_rf = tune_random_forest(X_train, y_train, X_test, y_test, le)
    save_model_objects(best_rf, scaler, le)

if __name__ == "__main__":
    main()
