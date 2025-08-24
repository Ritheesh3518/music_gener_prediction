from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def tune_random_forest(X_train, y_train, X_test, y_test, label_encoder):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    best_rf = grid.best_estimator_

    y_pred = best_rf.predict(X_test)
    print("\nTuned Random Forest Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    return best_rf
