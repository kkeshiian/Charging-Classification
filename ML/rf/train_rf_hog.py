import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

DATASET_PATH = "../dataset_slot"

def extract_hog(img):
    img = cv2.resize(img, (64, 64))
    return hog(img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys').astype(np.float32)

def load_data():
    X, y = [], []
    for label in os.listdir(DATASET_PATH):
        path = os.path.join(DATASET_PATH, label)
        if not os.path.isdir(path): continue
        for f in os.listdir(path):
            img = cv2.imread(os.path.join(path, f), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            X.append(extract_hog(img))
            y.append(label)
    return np.array(X), np.array(y)

X, y = load_data()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

train_accs, test_accs = [], []
all_y_test, all_y_pred = [], []

for tr, ts in kf.split(X):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X[tr], y[tr])

    y_tr = model.predict(X[tr])
    y_ts = model.predict(X[ts])

    train_accs.append(accuracy_score(y[tr], y_tr))
    test_accs.append(accuracy_score(y[ts], y_ts))

    all_y_test.extend(y[ts])
    all_y_pred.extend(y_ts)

print("Train:", np.mean(train_accs))
print("Test:", np.mean(test_accs))
print(classification_report(all_y_test, all_y_pred))

cm = confusion_matrix(all_y_test, all_y_pred)
ConfusionMatrixDisplay(cm).plot(cmap="Blues")
os.makedirs("matrix", exist_ok=True)
plt.savefig("matrix/cm_rf_hog.png")
plt.close()

joblib.dump(RandomForestClassifier().fit(X, y), "model/model_rf_hog.pkl")