# Face Authentication using Facenet

This project implements a face authentication system using the Facenet model for feature extraction and a neural network classifier for authentication. The system evaluates the effectiveness of face recognition in payment systems across POS devices and mobile platforms.

## Features
- Loads and balances the CelebA dataset for face authentication.
- Uses Facenet to extract deep face embeddings.
- Implements a neural network model for authentication.
- Evaluates performance using accuracy, FAR (False Acceptance Rate), and FRR (False Rejection Rate).
- Utilizes data augmentation techniques like SMOTE for handling class imbalance.

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install tensorflow numpy pandas matplotlib seaborn tqdm opencv-python deepface scikit-learn imbalanced-learn
```

Additionally, make sure you have the CelebA dataset downloaded and extracted at the specified path.

## Dataset
The dataset used in this project is [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Place the dataset in the following structure:
```
/home/user/Documents/Research/Datasets/CelebA/
├── identity_CelebA.txt
├── img_align_celeba/
    ├── 000001.jpg
    ├── 000002.jpg
    ├── ...
```

## Configuration
Modify the configuration parameters in the script as needed:
```python
CELEBA_DIR = "/home/user/Documents/Research/Datasets/CelebA"
IMG_SIZE = (160, 160)
BATCH_SIZE = 64
AUTHORIZED_IDS = [2880, 2937]  # Modify with actual person IDs
MIN_SAMPLES_PER_ID = 5
IMPOSTOR_SAMPLES = 1000
TEST_SIZE = 0.2
SEED = 42
```

## Running the Code
To execute the authentication pipeline, run:
```bash
python face_auth.py
```
This will:
1. Load and balance the dataset.
2. Extract facial features using Facenet.
3. Train a neural network for authentication.
4. Evaluate performance and generate metrics.

## Evaluation Metrics
The model evaluates authentication performance using:
- **Accuracy**: Measures overall correctness of authentication.
- **FAR (False Acceptance Rate)**: Measures unauthorized access.
- **FRR (False Rejection Rate)**: Measures rejection of authorized users.

The script also generates:
- **Confusion Matrix**
- **ROC Curve**
- **Precision-Recall Curve**

## Results and Visualization
After training, the model outputs key metrics and visualizations. Example output:
```
=== Final Report ===
Optimal Threshold: 0.5674
Accuracy: 0.9253
FAR: 0.0342
FRR: 0.0421
```
Plots are displayed for further analysis.

## License
This project is for research and educational purposes only.

---
For improvements, contributions, or issues, feel free to submit a pull request or open an issue on GitHub.

