import os
import cv2
import joblib
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import models
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

DATASET_PATH = "../dataset_slot"
IMG_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# CNN FEATURE EXTRACTOR
# ===============================
vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:10]
vgg = vgg.to(device)
vgg.eval()

pool = torch.nn.AdaptiveAvgPool2d((4, 4))

def image_to_tensor(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor.float().to(device)

def extract_cnn_feature(img):
    tensor = image_to_tensor(img)

    with torch.no_grad():
        feat = vgg(tensor)
        feat = pool(feat)

    return feat.cpu().numpy().flatten().astype(np.float32)

def extract_canny_cnn(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # global feature dari gambar asli
    global_feature = extract_cnn_feature(img)

    # edge image dari Canny
    edge = cv2.Canny(img, 50, 150)

    # deep edge feature dari CNN
    edge_feature = extract_cnn_feature(edge)

    # gabungan fitur
    feature = np.concatenate([global_feature, edge_feature])

    return feature.astype(np.float32)

# ===============================
# LOAD DATA
# ===============================
def load_data():
    X = []
    y = []

    for label in os.listdir(DATASET_PATH):
        label_path = os.path.join(DATASET_PATH, label)

        if not os.path.isdir(label_path):
            continue

        for fname in os.listdir(label_path):
            fpath = os.path.join(label_path, fname)

            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            feature = extract_canny_cnn(img)

            X.append(feature)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y)

# ===============================
# HIK KERNEL
# ===============================
def hik_kernel(X, Y):
    K = np.zeros((X.shape[0], Y.shape[0]), dtype=np.float32)
    for i in range(X.shape[0]):
        K[i] = np.sum(np.minimum(X[i], Y), axis=1)
    return K

# ===============================
# TRAINING
# ===============================
X, y = load_data()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

train_accuracies = []
test_accuracies = []

all_y_test = []
all_y_pred = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = SVC(kernel=hik_kernel)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    all_y_test.extend(y_test)
    all_y_pred.extend(y_test_pred)

all_y_test = np.array(all_y_test)
all_y_pred = np.array(all_y_pred)

print("Train Accuracy (avg):", np.mean(train_accuracies))
print("Test Accuracy (avg):", np.mean(test_accuracies))

print("\nClassification Report:")
print(classification_report(all_y_test, all_y_pred))

# ===============================
# CONFUSION MATRIX
# ===============================
labels = np.unique(y)
cm = confusion_matrix(all_y_test, all_y_pred, labels=labels)

os.makedirs("matrix", exist_ok=True)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix SVM Canny-CNN HIK")
plt.savefig("matrix/cm_svm_canny_cnn_hik.png")
plt.close()

# ===============================
# SAVE MODEL
# ===============================
final_model = SVC(kernel=hik_kernel)
final_model.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump(final_model, "model/model_svm_canny_cnn_hik.pkl")