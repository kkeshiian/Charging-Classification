# 🔌 Deteksi Slot Charger Kosong/Terisi Berbasis Citra

Sistem deteksi otomatis untuk mengidentifikasi **slot pengisian daya (charging station) ponsel** dalam kondisi **kosong** atau **terisi** menggunakan pengolahan citra dan *machine learning*. Proyek ini membandingkan pendekatan **Machine Learning klasik** (ekstraksi fitur manual + classifier) dengan **Deep Learning** (*transfer learning*).

---

## 📌 Penjelasan Singkat Proyek

Proyek ini mengklasifikasikan citra slot charger ke dalam dua kelas — **kosong** dan **terisi** — sebagai dasar sistem *monitoring* ketersediaan slot. Dua pendekatan diuji dan dibandingkan:

* **Machine Learning (UTS):** ekstraksi fitur manual (HOG, GLCM, Canny-CNN) + classifier (SVM, Random Forest, Gradient Boosting, Decision Tree). SVM diuji dengan kernel HIK dan RBF.
* **Deep Learning (UAS):** *transfer learning* dengan 4 arsitektur — **MobileNetV3, VGG16, ResNet50, ConvNeXt-Tiny** — pada skema klasifikasi biner.

---

## ✨ Fitur

* Klasifikasi biner: `kosong` vs `terisi`
* Pipeline Machine Learning dan Deep Learning dalam satu repository
* Evaluasi menggunakan **K-Fold / Stratified 5-Fold Cross Validation**
* Perbandingan performa antar-metode
* Penyimpanan model dan hasil evaluasi secara otomatis

---

## 📂 Dataset

* **Jumlah kelas:** 2 (`kosong`, `terisi`)
* **Ukuran input:** 256 × 256 piksel
* **Link dataset:** `<TEMPEL_LINK_DATASET_DI_SINI>`

Struktur folder dataset:

```text
dataset/
├── kosong/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── terisi/
    ├── img001.jpg
    ├── img002.jpg
    └── ...
```

Keterangan:

* `kosong` : slot charger tidak digunakan.
* `terisi` : slot charger sedang digunakan.

---

## ▶️ Cara Menjalankan Program

### 1. Install Dependensi

```bash
pip install -r requirements.txt
```

---

### 2. Siapkan Dataset

Pastikan dataset mengikuti struktur folder yang telah dijelaskan sebelumnya.

---

### 3. Machine Learning (UTS)

Contoh struktur:

```text
src/
├── svm_hog_hik.py
├── svm_glcm_hik.py
├── svm_canny_cnn_hik.py
├── random_forest_hog.py
├── gradient_boosting_hog.py
└── ...
```

Untuk menjalankan model, cukup eksekusi script yang diinginkan:

```bash
python src/svm_hog_hik.py
```

atau

```bash
python src/svm_glcm_hik.py
```

atau

```bash
python src/svm_canny_cnn_hik.py
```

atau model lainnya sesuai nama file.

---

### 4. Deep Learning (UAS)

Buka notebook dan jalankan seluruh sel secara berurutan:

```bash
jupyter notebook notebooks/02_deep_learning.ipynb
```

---

## 📁 Struktur Repository

```text
project/

├── dataset/
│   ├── kosong/
│   └── terisi/
│
├── ML/
│   ├── svm_hog_hik.py
│   ├── svm_glcm_hik.py
│   └── svm_canny_cnn_hik.py
│   
│
├── DL/
│   └── deep_learning.ipynb
│
├── results/
│
├── requirements.txt
│
└── README.md
```

---



---

## 📈 Metode yang Digunakan

### Machine Learning

**Feature Extraction**

* HOG (Histogram of Oriented Gradients)
* GLCM (Gray Level Co-occurrence Matrix)
* Canny + CNN Feature Extractor

**Classifier**

* SVM (Histogram Intersection Kernel / HIK)
* SVM (RBF Kernel)
* Random Forest
* Gradient Boosting
* Decision Tree

### Deep Learning

**Transfer Learning**

* MobileNetV3
* VGG16
* ResNet50
* ConvNeXt-Tiny

---

## 🏆 Hasil Utama

Perbandingan model terbaik yang diperoleh pada proyek ini:

| Pendekatan       | Model                 | Akurasi    | F1-Score   |
| ---------------- | --------------------- | ---------- | ---------- |
| Machine Learning | SVM + Canny-CNN + HIK | **99,63%** | **1,00**   |
| Machine Learning | SVM + HOG + HIK       | 99,26%     | 0,99       |
| Deep Learning    | **ResNet50**          | **99,82%** | **99,82%** |
| Deep Learning    | MobileNetV3           | 99,36%     | 99,36%     |
| Deep Learning    | ConvNeXt-Tiny         | 99,36%     | 99,36%     |
| Deep Learning    | VGG16                 | 97,44%     | 97,38%     |

### 🥇 Model Terbaik

Model dengan performa terbaik adalah **ResNet50** yang mencapai:

* Akurasi: **99,82%**
* F1-Score: **99,82%**
* RMSE: **0,0506**
* MAE: **0,0153**

Hasil tersebut menunjukkan bahwa pendekatan **Deep Learning berbasis Transfer Learning** memberikan performa sedikit lebih baik dibandingkan pendekatan **Machine Learning klasik**, meskipun keduanya sama-sama menghasilkan akurasi di atas 99%.
