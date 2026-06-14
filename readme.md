# 🔌 Deteksi Slot Charger Kosong/Terisi Berbasis Citra

Sistem deteksi otomatis untuk mengidentifikasi **slot pengisian daya (charging station) ponsel** dalam kondisi **kosong** atau **terisi** menggunakan pengolahan citra dan *machine learning*. Proyek ini membandingkan pendekatan **Machine Learning klasik** (ekstraksi fitur manual + classifier) dengan **Deep Learning** (*transfer learning*).

---

## 📌 Penjelasan Singkat Proyek
Proyek ini mengklasifikasikan citra slot charger ke dalam dua kelas — **kosong** dan **terisi** — sebagai dasar sistem *monitoring* ketersediaan slot. Dua pendekatan diuji dan dibandingkan:

- **Machine Learning (UTS):** ekstraksi fitur manual (HOG, GLCM, Canny-CNN) + classifier (SVM, Random Forest, Gradient Boosting, Decision Tree). SVM diuji dengan kernel HIK dan RBF.
- **Deep Learning (UAS):** *transfer learning* dengan 4 arsitektur — **MobileNetV3, VGG16, ResNet50, ConvNeXt-Tiny** — pada skema klasifikasi biner.

---

## ✨ Fitur
- Klasifikasi biner: `kosong` vs `terisi`
- Pipeline ML klasik & Deep Learning dalam satu repo
- Evaluasi dengan **K-Fold / Stratified 5-Fold Cross Validation**
- Perbandingan performa antar-metode (akurasi, F1, RMSE, MAE)

---

## 📂 Dataset
- **Jumlah kelas:** 2 (`kosong`, `terisi`)
- **Ukuran input:** 256 × 256 piksel
- **Link dataset:** <TEMPEL_LINK_DATASET_DI_SINI>  *(Google Drive / Kaggle / Roboflow)*

Struktur folder dataset: