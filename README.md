# 📬 Spam Email Classifier using Logistic Regression

## Email and Messaging
Di era digital saat ini, komunikasi melalui email menjadi bagian penting dari kehidupan sehari-hari, baik dalam konteks pribadi maupun profesional. Namun, meningkatnya volume email juga diiringi oleh ancaman seperti spam, phishing, dan pesan tidak diinginkan lainnya. Deteksi dini terhadap pesan-pesan ini sangat penting untuk menjaga keamanan dan kenyamanan pengguna.

Proyek ini bertujuan untuk membangun sistem klasifikasi email spam berbasis Machine Learning dan Natural Language Processing (NLP). Dengan menggunakan dataset berlabel spam dan ham, pesan teks diolah melalui teknik TF-IDF vectorization, lalu diklasifikasikan menggunakan algoritma Logistic Regression.

Model yang dihasilkan mampu membedakan pesan spam dari pesan normal (ham) dengan akurasi yang tinggi. Proyek ini mencakup seluruh pipeline, mulai dari:

- Eksplorasi dan pembersihan data,

- Ekstraksi fitur dari teks,

- Pelatihan dan evaluasi model,

- Penyimpanan model untuk digunakan kembali (inference),

- Hingga tahap akhir berupa antarmuka pengguna atau integrasi ke sistem lain.


## Data Understanding

Bagian ini menyajikan detail mengenai data yang digunakan dalam proyek klasifikasi ini, yaitu SMS Spam Collection Dataset. Dataset ini merupakan salah satu dataset klasik yang banyak digunakan dalam penelitian machine learning untuk masalah klasifikasi teks biner, khususnya dalam mendeteksi pesan spam.

Fokus dari proyek ini adalah membangun model yang mampu membedakan antara pesan spam (tidak diinginkan) dan ham (pesan biasa) berdasarkan isi teks pesan.

Dataset ini tersedia secara publik dan dapat diunduh dari berbagai sumber, salah satunya dari UCI Machine Learning Repository dan Kaggle:

https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification

Struktur Dataset

Dataset terdiri dari dua kolom utama:

| Kolom      | Deskripsi                                                                  |
|------------|----------------------------------------------------------------------------|
| `Category` | Label untuk pesan, terdiri dari dua nilai: `spam` dan `ham`.               |
| `Message`  | Isi pesan teks yang dikirim melalui SMS.                                   |

## 🧠 Karakteristik Data

- **Jumlah data**: ± 5.500 baris
- **Jenis tugas**: *Binary classification*
- **Target/label**: `Category` (`spam` atau `ham`)
- **Fitur utama**: Teks dari kolom `Message` yang diubah menjadi fitur numerik menggunakan teknik **TF-IDF (Term Frequency-Inverse Document Frequency)**.

# 📈 Logistic Regression for Spam Classification

## 🧠 Konsep Dasar Logistic Regression

Logistic Regression adalah algoritma klasifikasi biner yang banyak digunakan untuk memprediksi probabilitas dari dua kelas. Dalam konteks ini, model digunakan untuk mengklasifikasikan pesan sebagai **spam** atau **ham** (bukan spam).

### 🧮 Fungsi Aktivasi: Sigmoid
Model menghitung probabilitas sebuah pesan termasuk ke dalam kelas `spam (1)` menggunakan fungsi sigmoid:

\[
P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \ldots + \beta_n x_n)}}
\]

Jika hasil > 0.5 → diklasifikasikan sebagai spam (`1`), jika tidak → ham (`0`).

---
