import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from insightface.app import FaceAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import (confusion_matrix, accuracy_score, roc_auc_score,
                             classification_report, roc_curve, precision_recall_curve)
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
np.seterr(divide='ignore', invalid='ignore')

# Configure TensorFlow for GPU if available
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("TensorFlow configured to use GPU with memory growth")
    else:
        print("No GPU detected. Running on CPU.")
except Exception as e:
    print(f"Error during GPU setup: {e}")

# ===============================
# Configurations
# ===============================
CELEBA_DIR = "/home/vimal/Documents/CIS735/Research/CelebA"
IMG_SIZE = (112, 112)
BATCH_SIZE = 64
AUTHORIZED_IDS = [2880]
MIN_SAMPLES_PER_ID = 5
IMPOSTOR_SAMPLES = 4000
TEST_SIZE = 0.2
SEED = 42

def plot_training_history(history):
    plt.figure(figsize=(10,5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.3f})'.format(roc_auc_score(y_true, y_probs)))
    plt.plot([0,1],[0,1],'k--', label='Random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(cm, class_names=['Impostor', 'Authorized']):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# ===============================
# Load ArcFace Model
# ===============================
def load_arcface_model():
    print("Loading ArcFace model...")
    try:
        arcface_model = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        arcface_model.prepare(ctx_id=0, det_size=(640, 640))
        print("ArcFace model loaded successfully with CUDA provider")
    except Exception as e:
        print(f"Error loading buffalo_l model: {e}")
        try:
            arcface_model = FaceAnalysis(name="buffalo_s", providers=['CPUExecutionProvider'])
            arcface_model.prepare(ctx_id=-1, det_size=(640, 640))
            print("Fallback: ArcFace buffalo_s model loaded")
        except Exception as e2:
            print(f"Error loading fallback model: {e2}")
            raise
    return arcface_model

# ===============================
# Load Dataset
# ===============================
def load_dataset():
    identity_path = os.path.join(CELEBA_DIR, "identity_CelebA.txt")
    df = pd.read_csv(identity_path, sep=" ", header=None, names=["file", "person_id"])

    auth_df = df[df["person_id"].isin(AUTHORIZED_IDS)]
    auth_df = auth_df.groupby("person_id").filter(lambda x: len(x) >= MIN_SAMPLES_PER_ID)
    if auth_df.empty:
        raise ValueError(f"No authorized users with ≥{MIN_SAMPLES_PER_ID} samples.")

    impostor_df = df[~df["person_id"].isin(AUTHORIZED_IDS)].sample(IMPOSTOR_SAMPLES, random_state=SEED)
    combined_df = pd.concat([auth_df, impostor_df]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    combined_df["label"] = combined_df["person_id"].isin(AUTHORIZED_IDS).astype(int)

    print("\nClass distribution:\n", combined_df["label"].value_counts())
    return combined_df

# ===============================
# Image Preprocessing
# ===============================
def preprocess_image(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None or len(img.shape) != 3 or img.shape[2] != 3:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    except Exception:
        return None

# ===============================
# Load and preprocess images
# ===============================
def load_and_preprocess_images(df):
    images = []
    skipped_images = 0

    for img_file in tqdm(df["file"], desc="Loading images"):
        path = os.path.join(CELEBA_DIR, "img_align_celeba", img_file)
        img = preprocess_image(path)
        if img is not None:
            images.append(img)
        else:
            skipped_images += 1

    print(f"Successfully loaded {len(images)} images, skipped {skipped_images} images")
    return np.array(images)

# ===============================
# Extract ArcFace Embeddings
# ===============================
def extract_embeddings(arcface_model, images):
    embeddings = []
    failed_extractions = 0

    for img in tqdm(images, desc="Extracting ArcFace embeddings"):
        try:
            img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)
            faces = arcface_model.get(img)
            if faces and faces[0].embedding is not None and len(faces[0].embedding) == 512:
                emb = faces[0].embedding
                embeddings.append(emb / np.linalg.norm(emb))
            else:
                embeddings.append(np.zeros(512))
                failed_extractions += 1
        except Exception:
            embeddings.append(np.zeros(512))
            failed_extractions += 1

    print(f"Successfully extracted {len(embeddings) - failed_extractions} embeddings, failed on {failed_extractions} images")
    return np.array(embeddings)

# ===============================
# Neural Network Classifier
# ===============================
def create_nn_classifier(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ===============================
# Main
# ===============================
def main():
    try:
        arcface_model = load_arcface_model()
        df = load_dataset()
        images = load_and_preprocess_images(df)
        if len(images) == 0:
            raise ValueError("No images were successfully loaded")

        labels = df["label"].values[:len(images)]
        X_train_img, X_test_img, y_train, y_test = train_test_split(
            images, labels, test_size=TEST_SIZE, stratify=labels, random_state=SEED
        )

        print("Extracting embeddings for training set...")
        X_train_emb = extract_embeddings(arcface_model, X_train_img)
        print("Extracting embeddings for test set...")
        X_test_emb = extract_embeddings(arcface_model, X_test_img)

        valid_train_mask = np.any(X_train_emb != 0, axis=1)
        valid_test_mask = np.any(X_test_emb != 0, axis=1)

        X_train_emb = X_train_emb[valid_train_mask]
        y_train = y_train[valid_train_mask]
        X_test_emb = X_test_emb[valid_test_mask]
        y_test = y_test[valid_test_mask]

        scaler = StandardScaler()
        X_train_emb = scaler.fit_transform(X_train_emb)
        X_test_emb = scaler.transform(X_test_emb)

        model = create_nn_classifier(X_train_emb.shape[1])
        csv_logger = CSVLogger('training_log.csv', append=False)

        history = model.fit(
            X_train_emb, y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            class_weight={0: 1, 1: 20},
            verbose=1,
            callbacks=[csv_logger]
        )

        print("\nEpoch history:")
        for i, acc in enumerate(history.history['accuracy'], 1):
            val_acc = history.history['val_accuracy'][i - 1]
            print(f"Epoch {i}: train acc = {acc:.4f}, val acc = {val_acc:.4f}")

        nn_probs = model.predict(X_test_emb).flatten()
        fpr, tpr, thresholds = roc_curve(y_test, nn_probs)
        nn_threshold = thresholds[np.argmax(tpr)]  # Maximize recall (lower FRR)
        nn_pred = (nn_probs >= nn_threshold).astype(int)

        authorized_mask = y_train == 1
        oc_svm = OneClassSVM(kernel='rbf', nu=0.01, gamma='scale')
        oc_svm.fit(X_train_emb[authorized_mask])
        oc_pred = (oc_svm.predict(X_test_emb) == 1).astype(int)

        ensemble_pred = np.logical_or(nn_pred == 1, oc_pred == 1).astype(int)

        acc = accuracy_score(y_test, ensemble_pred) * 100
        cm = confusion_matrix(y_test, ensemble_pred)
        auc = roc_auc_score(y_test, nn_probs) if len(np.unique(y_test)) > 1 else 0.0
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
        frr = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0

        print(f"\nEnsemble Accuracy: {acc:.2f}%")
        print(f"AUC Score: {auc:.3f}")
        print(f"False Acceptance Rate (FAR): {far:.2f}%")
        print(f"False Rejection Rate (FRR): {frr:.2f}%")
        print("\nConfusion Matrix:\n", cm)
        print("\nClassification Report:\n", classification_report(y_test, ensemble_pred, zero_division=0))

        plot_training_history(history)
        plot_roc_curve(y_test, nn_probs)
        plot_confusion_matrix(cm)   

    except Exception as e:
        print(f"Error in main pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

