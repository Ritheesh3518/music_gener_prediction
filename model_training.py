from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=500),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'SVM': SVC(probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Gradient Boosting': GradientBoostingClassifier()
    }

def train_and_evaluate_models(models, X_train, X_test, y_train, y_test, label_encoder):
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name} trained.")

    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap="Blues")
        plt.title(f"{name} Confusion Matrix")
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.show()
