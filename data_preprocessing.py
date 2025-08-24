from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_features_labels(features, labels, test_size=0.2, random_state=42):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, y_encoded, test_size=test_size,
        random_state=random_state, stratify=y_encoded)

    return X_train, X_test, y_train, y_test, scaler, le
