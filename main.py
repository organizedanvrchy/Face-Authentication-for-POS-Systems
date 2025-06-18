import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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
                             classification_report, roc_curve)
import torch
import torch.nn as nn
import torch.optim as optim
import imgaug.augmenters as iaa
import warnings

# ===============================
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
np.seterr(divide='ignore', invalid='ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===============================
# Configurations
# ===============================
CELEBA_DIR = "/home/vimal/Documents/CIS735/Research/CelebA"
IMG_SIZE = (112, 112)
AUTHORIZED_IDS = [2880]
MIN_SAMPLES_PER_ID = 5
IMPOSTOR_SAMPLES = 4000
TEST_SIZE = 0.2
SEED = 42

# ===============================
# Data Augmentation (imgaug)
# ===============================
augmenter = iaa.Sequential([
    iaa.AdditiveGaussianNoise(scale=(10, 30)),
    iaa.GaussianBlur(sigma=(0.0, 2.0)),
    iaa.MotionBlur(k=5),
    iaa.Add((-30, 30)),
    iaa.Multiply((0.7, 1.3)),
    iaa.CoarseDropout(0.1, size_percent=0.2),
])

def augment_image(image):
    return augmenter(image=image)

def show_augmented_example(images, index=0):
    original = images[index]
    augmented = augment_image(original)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(augmented)
    axes[1].set_title("Augmented")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

# ===============================
# Plotting Functions
# ===============================
def plot_accuracy_history(train_accuracies, val_accuracies):
    plt.figure(figsize=(10,5))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_loss_history(train_losses, val_losses):
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.3f})'.format(roc_auc_score(y_true, y_probs)))
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(cm):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Impostor','Authorized'], yticklabels=['Impostor','Authorized'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# ===============================
# ArcFace Model Loading
# ===============================
def load_arcface_model():
    print("Loading ArcFace model...")
    try:
        arcface_model = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        arcface_model.prepare(ctx_id=0, det_size=(640, 640))
        print("ArcFace model loaded successfully with CUDA")
    except:
        arcface_model = FaceAnalysis(name="buffalo_s", providers=['CPUExecutionProvider'])
        arcface_model.prepare(ctx_id=-1, det_size=(640, 640))
        print("Fallback: ArcFace buffalo_s model loaded on CPU")
    return arcface_model

# ===============================
# Dataset Loading
# ===============================
def load_dataset():
    identity_path = os.path.join(CELEBA_DIR, "identity_CelebA.txt")
    df = pd.read_csv(identity_path, sep=" ", header=None, names=["file", "person_id"])
    auth_df = df[df["person_id"].isin(AUTHORIZED_IDS)]
    auth_df = auth_df.groupby("person_id").filter(lambda x: len(x) >= MIN_SAMPLES_PER_ID)
    if auth_df.empty:
        raise ValueError("No authorized users with sufficient samples.")
    impostor_df = df[~df["person_id"].isin(AUTHORIZED_IDS)].sample(IMPOSTOR_SAMPLES, random_state=SEED)
    combined_df = pd.concat([auth_df, impostor_df]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    combined_df['label'] = combined_df['person_id'].isin(AUTHORIZED_IDS).astype(int)
    print("Class distribution:\n", combined_df['label'].value_counts())
    return combined_df

# ===============================
# Image Preprocessing
# ===============================
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None or len(img.shape) != 3:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img.astype(np.uint8)

def load_and_preprocess_images(df):
    images, skipped = [], 0
    for img_file in tqdm(df['file'], desc="Loading images"):
        path = os.path.join(CELEBA_DIR, "img_align_celeba", img_file)
        img = preprocess_image(path)
        if img is not None:
            images.append(img)
        else:
            skipped += 1
    print(f"Loaded {len(images)}, skipped {skipped}")
    return np.array(images)

# ===============================
# ArcFace Embedding Extraction
# ===============================
def extract_embeddings(arcface_model, images):
    embeddings, failed = [], 0
    for img in tqdm(images, desc="Extracting embeddings"):
        try:
            faces = arcface_model.get(img)
            if faces and faces[0].embedding is not None:
                emb = faces[0].embedding / np.linalg.norm(faces[0].embedding)
                embeddings.append(emb)
            else:
                embeddings.append(np.zeros(512))
                failed += 1
        except:
            embeddings.append(np.zeros(512))
            failed += 1
    print(f"Extracted {len(embeddings)-failed} embeddings, failed: {failed}")
    return np.array(embeddings)

# ===============================
# PyTorch Classifier
# ===============================
class NNClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ===============================
# Training Loop
# ===============================
def train_model(model, train_loader, val_loader, epochs=20):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch).squeeze()
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                output = model(X_val).squeeze()
                loss = criterion(output, y_val)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))
        print(f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}  Val Loss={val_losses[-1]:.4f}")
    return train_losses, val_losses

# ===============================
# Main Pipeline
# ===============================
def main():
    arcface_model = load_arcface_model()
    df = load_dataset()
    images = load_and_preprocess_images(df)
    if len(images) == 0:
        raise ValueError("No images loaded.")

    show_augmented_example(images)
    labels = df['label'].values[:len(images)]

    X_train_img, X_test_img, y_train, y_test = train_test_split(
        images, labels, test_size=TEST_SIZE, stratify=labels, random_state=SEED)

    X_train_img_aug = np.array([augment_image(img) for img in X_train_img])

    print("Extracting embeddings for train...")
    X_train_emb = extract_embeddings(arcface_model, X_train_img_aug)
    print("Extracting embeddings for test...")
    X_test_emb = extract_embeddings(arcface_model, X_test_img)

    valid_train = np.any(X_train_emb != 0, axis=1)
    valid_test = np.any(X_test_emb != 0, axis=1)
    X_train_emb, y_train = X_train_emb[valid_train], y_train[valid_train]
    X_test_emb, y_test = X_test_emb[valid_test], y_test[valid_test]

    scaler = StandardScaler()
    X_train_emb = scaler.fit_transform(X_train_emb)
    X_test_emb = scaler.transform(X_test_emb)

    # PyTorch dataset
    X_train_tensor = torch.tensor(X_train_emb, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_emb, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_ds = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_split = int(len(train_ds)*0.8)
    train_set, val_set = torch.utils.data.random_split(train_ds, [val_split, len(train_ds)-val_split])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=32)

    model = NNClassifier(512).to(device)
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, train_loader, val_loader)

    model.eval()
    with torch.no_grad():
        probs = model(X_test_tensor.to(device)).cpu().squeeze().numpy()

    fpr, tpr, thresholds = roc_curve(y_test, probs)
    best_thresh = thresholds[np.argmax(tpr)]
    nn_pred = (probs >= best_thresh).astype(int)

    authorized_mask = y_train == 1
    oc_svm = OneClassSVM(kernel='rbf', nu=0.01, gamma='scale')
    oc_svm.fit(X_train_emb[authorized_mask])
    oc_pred = (oc_svm.predict(X_test_emb) == 1).astype(int)

    ensemble_pred = np.logical_or(nn_pred, oc_pred).astype(int)

    cm = confusion_matrix(y_test, ensemble_pred)
    acc = accuracy_score(y_test, ensemble_pred)*100
    auc = roc_auc_score(y_test, probs)
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp+tn) * 100
    frr = fn / (fn+tp) * 100

    print(f"\nEnsemble Accuracy: {acc:.2f}%")
    print(f"AUC: {auc:.3f}")
    print(f"FAR: {far:.2f}%  FRR: {frr:.2f}%")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, ensemble_pred))

    plot_loss_history(train_losses, val_losses)
    plot_accuracy_history(train_accuracies, val_accuracies)
    plot_roc_curve(y_test, probs)
    plot_confusion_matrix(cm)

if __name__ == "__main__":
    main()
    
