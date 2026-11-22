# Studi Komparatif Empiris Model Generatif & Batasan Model Klasifikasi

Repositori ini memuat implementasi komparatif empiris model generatif: **Vanilla Autoencoder**, **Sparse Autoencoder**, **Variational Autoencoder (VAE)**, **Vanilla GAN**, **Deep Convolutional GAN (DCGAN)**, dan **Denoising Diffusion Probabilistic Models (DDPM)** pada dataset **Fashion-MNIST**, serta modul eksperimen pembuktian batasan teoretis **MLP Classifier** untuk tugas *image generation*.

Proyek ini disusun untuk memenuhi tugas *Take-Home Assessment* mata kuliah **Deep Learning (D6421)** Sesi 07.

---

## 1. Struktur Repositori

```text
.
├── dataset.py               # Data loader Fashion-MNIST (split 80:10:10, seed 42)
├── models.py                # Implementasi PyTorch seluruh arsitektur model
├── exp_mlp_limits.py        # Eksperimen 5 tahap pembuktian batasan MLP Classifier
├── train_generative.py      # Pipeline pelatihan 6 model generatif
├── evaluate_and_plot.py     # Perhitungan metrik (MSE, PSNR, SSIM, FID, Latensi) & 5 figur wajib
├── main.py                  # CLI runner terintegrasi
├── requirements.txt         # Daftar pustaka dependensi
├── figures/                 # Direktori output 5 figur komparatif beresolusi tinggi (300 DPI)
└── outputs/                 # Log metrik numerik JSON dan kurva latih
```

---

## 2. Instalasi dan Setup Lingkungan

Pastikan Python 3.10+ dan PyTorch (dengan dukungan CUDA jika memiliki GPU) telah terpasang.

```bash
pip install -r requirements.txt
```

---

## 3. Petunjuk Eksekusi Eksperimen

Anda dapat menjalankan seluruh eksperimen secara langsung atau memilih tahap tertentu melalui argumen CLI `main.py`:

### Menjalankan Seluruh Eksperimen Sekaligus:
```bash
python main.py --step all --epochs 15
```

### Menjalankan Per Modul:
1. **Eksperimen Pembuktian Batasan MLP Classifier (Tahap 1–5):**
   ```bash
   python main.py --step mlp
   ```
2. **Pelatihan 6 Model Generatif:**
   ```bash
   python main.py --step train --epochs 15
   ```
3. **Evaluasi Metrik Komparatif & Ekspor 5 Figur Laporan:**
   ```bash
   python main.py --step eval
   ```

---

## 4. Rincian Metrik & Figur Laporan Ilmiah

Eksekusi skrip evaluasi akan menghasilkan:
* `outputs/tabel_komparasi_metrik.json`: Kompilasi metrik numerik lengkap (Training time/epoch, Latensi inferensi batch 64, Test MSE, Test PSNR, Test SSIM, FID Score).
* `figures/Figur_1_Rekonstruksi_SideBySide.png`: Rekonstruksi Ground Truth vs Vanilla AE vs Sparse AE vs VAE.
* `figures/Figur_2_Random_Generative_Sampling.png`: Kisi 4x4 sintesis generatif (VAE, Vanilla GAN, DCGAN, DDPM).
* `figures/Figur_3_Batasan_MLP.png`: Visualisasi pembuktian empiris ketidakmampuan bobot dan ruang laten MLP Classifier.
* `figures/Figur_4_Latent_Interpolation.png`: Transisi interpolasi linier $z(\alpha) = (1-\alpha)z_A + \alpha z_B$.
* `figures/Figur_5_LossCurves_and_DiffusionTrajectory.png`: Kurva stabilitas training dan trajektori denoising DDPM ($x_T \to x_0$).
