import os
import tempfile
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from deepface import DeepFace
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score, roc_auc_score,
                             classification_report, roc_curve, precision_recall_curve)

# ==============================
# Configuration
# ==============================
CELEBA_DIR = "/home/vimal/Documents/Research/Datasets/CelebA"
IMG_SIZE = (160, 160)  # Facenet input size
BATCH_SIZE = 64
AUTHORIZED_IDS = [2880]  # Actual person ID from identity file
MIN_SAMPLES_PER_ID = 5  # Minimum images per authorized user
IMPOSTOR_SAMPLES = 4000  # Number of impostor samples to use
TEST_SIZE = 0.2
SEED = 42

# ==============================
# Data Pipeline
# ==============================
def load_and_balance_dataset():
    """Load dataset with balanced classes"""
    identity_path = os.path.join(CELEBA_DIR, "identity_CelebA.txt")
    df = pd.read_csv(identity_path, sep=" ", header=None, names=["file", "person_id"])
    
    # Filter authorized users with enough samples
    auth_df = df[df["person_id"].isin(AUTHORIZED_IDS)]
    auth_df = auth_df.groupby("person_id").filter(lambda x: len(x) >= MIN_SAMPLES_PER_ID)
    
    if auth_df.empty:
        raise ValueError(f"No authorized users found with ≥{MIN_SAMPLES_PER_ID} samples")
    
    # Select impostors
    impostor_df = df[~df["person_id"].isin(AUTHORIZED_IDS)].sample(IMPOSTOR_SAMPLES)
    
    # Combine and shuffle
    balanced_df = pd.concat([auth_df, impostor_df]).sample(frac=1, random_state=SEED)
    balanced_df["label"] = balanced_df["person_id"].isin(AUTHORIZED_IDS).astype(int)
    
    print("\nClass distribution:")
    print(balanced_df["label"].value_counts())
    
    return balanced_df

# ==============================
# Feature Extraction with Facenet
# ==============================
def extract_facenet_features(images):
    """Extract features using Facenet"""
    features = []
    for img in tqdm(images, desc="Extracting features"):
        # Convert and save temp image
        img = (img * 255).astype(np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            cv2.imwrite(tmp_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
        try:
            embedding = DeepFace.represent(
                img_path=tmp_path,
                model_name="Facenet",  # Using Facenet
                enforce_detection=False,
                detector_backend="skip"
            )[0]["embedding"]
            features.append(embedding)
        finally:
            os.remove(tmp_path)
    
    return np.array(features)

# ==============================
# Model Architecture
# ==============================
def create_authentication_model(input_dim):
    """Create balanced model with regularization"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ==============================
# Training and Evaluation
# ==============================
def main():
    # # Force GPU usage
    # tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')

    # # Ensure memory growth is enabled
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     for gpu in gpus:
    #         tf.config.experimental.set_memory_growth(gpu, True)

    # # Check for GPU initialization
    # print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
    # print("GPU Device Name:", tf.test.gpu_device_name())

    # Load and prepare data
    print("\n=== Loading Dataset ===")
    df = load_and_balance_dataset()
    
    # Load images
    print("\n=== Loading Images ===")
    images, labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(CELEBA_DIR, "img_align_celeba", row["file"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE) / 255.0
        images.append(img)
        labels.append(row["label"])
    
    # Convert to NumPy arrays
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=TEST_SIZE, stratify=labels, random_state=SEED
    )
    
    # Extract features
    print("\n=== Feature Extraction ===")
    X_train = extract_facenet_features(X_train)
    X_test = extract_facenet_features(X_test)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Handle remaining imbalance
    smote = SMOTE(sampling_strategy=0.5, random_state=SEED)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Ensure NumPy arrays
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    
    # Build and train model
    print("\n=== Training Model ===")
    model = create_authentication_model(X_train.shape[1])
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=128,
        validation_split=0.2,
        class_weight={0: 1, 1: 5},  # Adjust for class imbalance
        callbacks=[early_stop],
        verbose=1
    )

    # Optimal threshold selection
    y_probs = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    optimal_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[optimal_idx]
    
    # Final predictions
    y_pred = (y_probs > best_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred) * 100
    cm = confusion_matrix(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_probs)
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Confusion Matrix
    plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Impostor", "Authorized"],
                yticklabels=["Impostor", "Authorized"])
    plt.title("Confusion Matrix")
    
    # ROC Curve with AUC
    plt.subplot(1, 3, 2)
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    
    # Precision-Recall Curve
    plt.subplot(1, 3, 3)
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    
    # Plot Accuracy Over Epochs
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.tight_layout()
    plt.show()
    
    # Print TP, TN, FP, FN in a table
    confusion_df = pd.DataFrame(
        [[tp, fp], [fn, tn]],
        columns=["Predicted Positive", "Predicted Negative"],
        index=["Actual Positive", "Actual Negative"]
    )
    print("\n=== Confusion Matrix Table ===")
    print(confusion_df)

    # Print performance metrics
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "FAR", "FRR"],
        "Value": [f"{accuracy:.4f}%", f"{far:.4f}%", f"{frr:.4f}%"]
    })
    print("\n=== Performance Metrics ===")
    print(metrics_df.to_string(index=False))

    # Classification Report Table
    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    class_report_df = pd.DataFrame(class_report).transpose()
    print("\n=== Classification Report ===")
    print(class_report_df.to_string())

if __name__ == "__main__":
    main()
