import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

DATASET_PATH = "../dataset_slot"

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

            img = cv2.resize(img, (64, 64))
            feature = img.flatten().astype(np.float32)

            X.append(feature)
            y.append(label)

    return np.array(X), np.array(y)

X, y = load_data()

X = X / 255.0

kf = KFold(n_splits=5, shuffle=True, random_state=42)

train_accuracies = []
test_accuracies = []

all_y_test = []
all_y_pred = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = SVC(kernel='rbf')
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

avg_train_acc = np.mean(train_accuracies)
avg_test_acc = np.mean(test_accuracies)

print("Train Accuracy (avg):", avg_train_acc)
print("Test Accuracy (avg):", avg_test_acc)

print("\nClassification Report:")
print(classification_report(all_y_test, all_y_pred))

labels = np.unique(y)
cm = confusion_matrix(all_y_test, all_y_pred, labels=labels)

os.makedirs("matrix", exist_ok=True)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix SVM RBF")
plt.savefig("matrix/cm_svm_rbf.png")
plt.close()

final_model = SVC(kernel='rbf')
final_model.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump(final_model, "model/model_svm_raw_rbf.pkl")