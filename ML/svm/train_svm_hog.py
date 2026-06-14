import time
import cv2
import joblib
import numpy as np
import streamlit as st
import tensorflow as tf
import torch

from PIL import Image
from skimage.feature import hog
from torchvision import models

# =========================================================

# KONFIGURASI

# =========================================================

st.set_page_config(
page_title="Klasifikasi Slot Charger EV",
page_icon="🔌",
layout="wide"
)

CLASS_NAMES = ["Kosong", "Terisi"]

DL_IMG_SIZE = (256, 256)

MODEL_PATHS = {
"SVM + HOG + HIK":
"models/model_svm_hog_hik.pkl",

```
"SVM + Canny CNN + HIK":
    "models/model_svm_canny_cnn_hik.pkl",

"MobileNetV3":
    "models/MobileNetV3_best.keras",

"VGG16":
    "models/VGG16_best.keras",

"ResNet50":
    "models/ResNet50_best.keras",

"ConvNeXt-Tiny":
    "models/ConvNeXtTiny_best.keras"
```

}

ML_MODELS = [
"SVM + HOG + HIK",
"SVM + Canny CNN + HIK"
]

DEVICE = torch.device(
"cuda" if torch.cuda.is_available()
else "cpu"
)

# =========================================================

# HISTOGRAM INTERSECTION KERNEL

# =========================================================

def hik_kernel(X, Y):

```
K = np.zeros(
    (X.shape[0], Y.shape[0]),
    dtype=np.float32
)

for i in range(X.shape[0]):
    K[i] = np.sum(
        np.minimum(X[i], Y),
        axis=1
    )

return K
```

# =========================================================

# HOG FEATURE

# =========================================================

def extract_hog_feature(image):

```
img = image.convert("L")
img = np.array(img)

img = cv2.resize(
    img,
    (64, 64)
)

feat = hog(
    img,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys"
)

return feat.astype(
    np.float32
).reshape(1, -1)
```

# =========================================================

# VGG16 FEATURE EXTRACTOR

# =========================================================

@st.cache_resource
def load_vgg_feature_extractor():

```
vgg = models.vgg16(
    weights=models.VGG16_Weights.IMAGENET1K_V1
).features[:10]

vgg = vgg.to(DEVICE)
vgg.eval()

pool = torch.nn.AdaptiveAvgPool2d((4, 4))

return vgg, pool
```

def image_to_tensor(img):

```
img = cv2.resize(
    img,
    (128, 128)
)

img = cv2.cvtColor(
    img,
    cv2.COLOR_GRAY2RGB
)

img = img.astype(
    np.float32
) / 255.0

mean = np.array(
    [0.485, 0.456, 0.406]
)

std = np.array(
    [0.229, 0.224, 0.225]
)

img = (img - mean) / std

tensor = torch.from_numpy(img)
tensor = tensor.permute(2, 0, 1)
tensor = tensor.unsqueeze(0)

return tensor.float().to(DEVICE)
```

def extract_cnn_feature(gray_img):

```
vgg, pool = load_vgg_feature_extractor()

tensor = image_to_tensor(
    gray_img
)

with torch.no_grad():

    feat = vgg(tensor)
    feat = pool(feat)

return feat.cpu().numpy().flatten()
```

def extract_canny_cnn_feature(image):

```
gray = image.convert("L")

gray = np.array(gray)

gray = cv2.resize(
    gray,
    (128, 128)
)

global_feat = extract_cnn_feature(
    gray
)

edge = cv2.Canny(
    gray,
    50,
    150
)

edge_feat = extract_cnn_feature(
    edge
)

feat = np.concatenate(
    [global_feat, edge_feat]
)

return feat.astype(
    np.float32
).reshape(1, -1)
```

# =========================================================

# LOAD MODEL

# =========================================================

@st.cache_resource
def load_model(model_name):

```
path = MODEL_PATHS[model_name]

if model_name in ML_MODELS:
    return joblib.load(path)

return tf.keras.models.load_model(
    path
)
```

# =========================================================

# PREPROCESS DEEP LEARNING

# =========================================================

def preprocess_dl(image):

```
image = image.convert("RGB")

image = image.resize(
    DL_IMG_SIZE
)

arr = np.array(
    image,
    dtype=np.float32
)

arr = np.expand_dims(
    arr,
    axis=0
)

return arr
```

# =========================================================

# PREDIKSI SVM

# =========================================================

def predict_svm(
model,
model_name,
image
):

```
if model_name == "SVM + HOG + HIK":

    feat = extract_hog_feature(
        image
    )

else:

    feat = extract_canny_cnn_feature(
        image
    )

pred = model.predict(
    feat
)[0]

label = int(pred)

return label, 1.0
```

# =========================================================

# PREDIKSI DL

# =========================================================

def predict_dl(
model,
image
):

```
x = preprocess_dl(
    image
)

prob = float(
    model.predict(
        x,
        verbose=0
    )[0][0]
)

label = (
    1 if prob >= 0.5
    else 0
)

confidence = (
    prob if label == 1
    else (1 - prob)
)

return (
    label,
    confidence,
    prob
)
```

# =========================================================

# UI

# =========================================================

st.title(
"🔌 Klasifikasi Kondisi Slot Charger EV"
)

st.markdown(
"""
Upload gambar slot charger untuk
menentukan kondisi:

```
- Kosong
- Terisi
"""
```

)

with st.sidebar:

```
st.header("Pengaturan")

selected_model = st.selectbox(
    "Pilih Model",
    list(MODEL_PATHS.keys())
)
```

uploaded = st.file_uploader(
"Upload Gambar",
type=[
"jpg",
"jpeg",
"png"
]
)

if uploaded is not None:

```
image = Image.open(uploaded)

col1, col2 = st.columns(
    [1, 1]
)

with col1:

    st.image(
        image,
        caption="Gambar Input",
        use_container_width=True
    )

with col2:

    if st.button(
        "Prediksi",
        use_container_width=True
    ):

        start = time.time()

        with st.spinner(
            "Memuat model..."
        ):

            model = load_model(
                selected_model
            )

        with st.spinner(
            "Melakukan prediksi..."
        ):

            if selected_model in ML_MODELS:

                label, confidence = (
                    predict_svm(
                        model,
                        selected_model,
                        image
                    )
                )

                prob = (
                    confidence
                    if label == 1
                    else 1 - confidence
                )

            else:

                (
                    label,
                    confidence,
                    prob
                ) = predict_dl(
                    model,
                    image
                )

        elapsed = (
            time.time() - start
        )

        st.success(
            f"Hasil Prediksi: "
            f"{CLASS_NAMES[label]}"
        )

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

        st.progress(
            float(confidence)
        )

        st.write(
            f"Probabilitas Terisi : "
            f"{prob*100:.2f}%"
        )

        st.write(
            f"Probabilitas Kosong : "
            f"{(1-prob)*100:.2f}%"
        )

        st.info(
            f"Waktu Inferensi: "
            f"{elapsed:.3f} detik"
        )

        st.caption(
            f"Model: "
            f"{selected_model}"
        )
```

else:

```
st.info(
    "Silakan upload gambar terlebih dahulu."
)
```
