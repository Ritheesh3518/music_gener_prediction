import joblib

def save_model_objects(model, scaler, label_encoder,
                       model_path="model.joblib", scaler_path="scaler.joblib", le_path="label_encoder.joblib"):
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    joblib.dump(label_encoder, le_path)
    print(f"Label encoder saved to {le_path}")
