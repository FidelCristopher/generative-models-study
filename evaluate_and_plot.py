import os
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import linalg
from skimage.metrics import structural_similarity as ssim

from dataset import get_dataloaders
from models import (
    VanillaAE, VAE, VanillaGenerator, DCGANGenerator,
    SimpleUNet, DDPM, MLPClassifier
)

def compute_psnr(mse):
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)

def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if isinstance(covmean, tuple):
            covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def run_evaluation_and_figures(device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Device: {device}")
    os.makedirs("./figures", exist_ok=True)
    os.makedirs("./outputs", exist_ok=True)

    _, _, test_loader = get_dataloaders(batch_size=128, seed=42)

    # Feature extractor untuk FID
    feature_extractor = MLPClassifier().to(device)
    if os.path.exists("./checkpoints/mlp_classifier.pt"):
        feature_extractor.load_state_dict(torch.load("./checkpoints/mlp_classifier.pt", map_location=device))
    feature_extractor.eval()

    real_features = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            feats = feature_extractor.forward_features(x)
            real_features.append(feats.cpu().numpy())
    real_features = np.concatenate(real_features, axis=0)
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)

    # Load model
    vanilla_ae = VanillaAE(latent_dim=64).to(device)
    sparse_ae = VanillaAE(latent_dim=64).to(device)
    vae = VAE(latent_dim=64).to(device)
    v_netG = VanillaGenerator(latent_dim=64).to(device)
    dc_netG = DCGANGenerator(latent_dim=100).to(device)
    unet = SimpleUNet(in_channels=1, base_channels=32).to(device)
    ddpm = DDPM(network=unet, timesteps=300, device=device)

    if os.path.exists("./checkpoints/vanilla_ae.pt"):
        vanilla_ae.load_state_dict(torch.load("./checkpoints/vanilla_ae.pt", map_location=device))
    if os.path.exists("./checkpoints/sparse_ae.pt"):
        sparse_ae.load_state_dict(torch.load("./checkpoints/sparse_ae.pt", map_location=device))
    if os.path.exists("./checkpoints/vae.pt"):
        vae.load_state_dict(torch.load("./checkpoints/vae.pt", map_location=device))
    if os.path.exists("./checkpoints/vanilla_gan_g.pt"):
        v_netG.load_state_dict(torch.load("./checkpoints/vanilla_gan_g.pt", map_location=device))
    if os.path.exists("./checkpoints/dcgan_g.pt"):
        dc_netG.load_state_dict(torch.load("./checkpoints/dcgan_g.pt", map_location=device))
    if os.path.exists("./checkpoints/ddpm_unet.pt"):
        unet.load_state_dict(torch.load("./checkpoints/ddpm_unet.pt", map_location=device))

    vanilla_ae.eval()
    sparse_ae.eval()
    vae.eval()
    v_netG.eval()
    dc_netG.eval()
    unet.eval()

    history = {}
    if os.path.exists("./outputs/loss_history.json"):
        with open("./outputs/loss_history.json", "r") as f:
            history = json.load(f)

    # Pengukuran latensi
    def measure_latency(func, *args, warmups=5, runs=20):
        for _ in range(warmups):
            _ = func(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = func(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return ((time.perf_counter() - t0) / runs) * 1000.0

    dummy_z64 = torch.randn(64, 64, device=device)
    dummy_z100 = torch.randn(64, 100, device=device)

    lat_vae = measure_latency(vanilla_ae.decode, dummy_z64)
    lat_sae = measure_latency(sparse_ae.decode, dummy_z64)
    lat_vae_gen = measure_latency(vae.decode, dummy_z64)
    lat_vgan = measure_latency(v_netG, dummy_z64)
    lat_dcgan = measure_latency(dc_netG, dummy_z100)
    lat_ddpm = measure_latency(ddpm.sample, (64, 1, 28, 28), warmups=0, runs=1)

    print("Evaluasi metrik rekonstruksi test set...")
    def eval_recon(model, is_vae=False):
        mses, ssims = [], []
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                if is_vae:
                    recon, _, _, _ = model(x)
                else:
                    recon, _ = model(x)
                mse_val = F.mse_loss(recon, x).item()
                mses.append(mse_val)

                for orig, rec in zip(x.cpu().numpy()[:10], recon.cpu().numpy()[:10]):
                    ssims.append(ssim(orig[0], rec[0], data_range=1.0))
        mean_mse = np.mean(mses)
        mean_psnr = compute_psnr(mean_mse)
        mean_ssim = np.mean(ssims)
        return mean_mse, mean_psnr, mean_ssim

    ae_mse, ae_psnr, ae_ssim = eval_recon(vanilla_ae)
    sae_mse, sae_psnr, sae_ssim = eval_recon(sparse_ae)
    vae_mse, vae_psnr, vae_ssim = eval_recon(vae, is_vae=True)

    print("Evaluasi skor FID...")
    def eval_fid(generator_fn, is_ddpm=False, latent_dim=64, num_samples=1000):
        feats = []
        bs = 100
        with torch.no_grad():
            for _ in range(num_samples // bs):
                if is_ddpm:
                    fake_x = generator_fn((bs, 1, 28, 28))
                else:
                    z = torch.randn(bs, latent_dim, device=device)
                    fake_x = generator_fn(z)
                f = feature_extractor.forward_features(fake_x)
                feats.append(f.cpu().numpy())
        feats = np.concatenate(feats, axis=0)
        mu_fake, sigma_fake = np.mean(feats, axis=0), np.cov(feats, rowvar=False)
        return compute_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    fid_vae = eval_fid(vae.decode, latent_dim=64, num_samples=500)
    fid_vgan = eval_fid(v_netG, latent_dim=64, num_samples=500)
    fid_dcgan = eval_fid(dc_netG, latent_dim=100, num_samples=500)
    fid_ddpm = eval_fid(ddpm.sample, is_ddpm=True, num_samples=100)

    table_data = [
        {"Model": "Vanilla Autoencoder", "Dim Laten": "64", "Train Time/Epoch (s)": round(history.get("vanilla_ae", {}).get("avg_time_epoch", 0.0), 2), "Latency (ms/64)": round(lat_vae, 2), "MSE": round(ae_mse, 5), "PSNR": round(ae_psnr, 2), "SSIM": round(ae_ssim, 4), "FID": "N/A (Latent Voids)", "Kestabilan": "Stabil / Konvergen"},
        {"Model": "Sparse Autoencoder", "Dim Laten": "64", "Train Time/Epoch (s)": round(history.get("sparse_ae", {}).get("avg_time_epoch", 0.0), 2), "Latency (ms/64)": round(lat_sae, 2), "MSE": round(sae_mse, 5), "PSNR": round(sae_psnr, 2), "SSIM": round(sae_ssim, 4), "FID": "N/A (Latent Voids)", "Kestabilan": "Stabil / Konvergen"},
        {"Model": "Variational Autoencoder (VAE)", "Dim Laten": "64", "Train Time/Epoch (s)": round(history.get("vae", {}).get("avg_time_epoch", 0.0), 2), "Latency (ms/64)": round(lat_vae_gen, 2), "MSE": round(vae_mse, 5), "PSNR": round(vae_psnr, 2), "SSIM": round(vae_ssim, 4), "FID": round(fid_vae, 2), "Kestabilan": "Stabil / Konvergen"},
        {"Model": "Vanilla GAN", "Dim Laten": "64", "Train Time/Epoch (s)": round(history.get("vanilla_gan", {}).get("avg_time_epoch", 0.0), 2), "Latency (ms/64)": round(lat_vgan, 2), "MSE": "N/A", "PSNR": "N/A", "SSIM": "N/A", "FID": round(fid_vgan, 2), "Kestabilan": "Rentan Osilasi / Mode Collapse"},
        {"Model": "DCGAN", "Dim Laten": "100", "Train Time/Epoch (s)": round(history.get("dcgan", {}).get("avg_time_epoch", 0.0), 2), "Latency (ms/64)": round(lat_dcgan, 2), "MSE": "N/A", "PSNR": "N/A", "SSIM": "N/A", "FID": round(fid_dcgan, 2), "Kestabilan": "Cukup Stabil"},
        {"Model": "DDPM", "Dim Laten": "28x28", "Train Time/Epoch (s)": round(history.get("ddpm", {}).get("avg_time_epoch", 0.0), 2), "Latency (ms/64)": round(lat_ddpm, 2), "MSE": "N/A", "PSNR": "N/A", "SSIM": "N/A", "FID": round(fid_ddpm, 2), "Kestabilan": "Sangat Stabil"}
    ]

    with open("./outputs/tabel_komparasi_metrik.json", "w") as f:
        json.dump(table_data, f, indent=4)

    # Figur 1: Side by side reconstruction
    print("Menyimpan Figur 1...")
    test_batch, _ = next(iter(test_loader))
    test_batch = test_batch[:8].to(device)

    with torch.no_grad():
        recon_vae, _ = vanilla_ae(test_batch)
        recon_sae, _ = sparse_ae(test_batch)
        recon_v, _, _, _ = vae(test_batch)

    fig, axes = plt.subplots(4, 8, figsize=(14, 7))
    rows = [
        ("Citra Asli (Ground Truth)", test_batch.cpu()),
        ("Rekonstruksi Vanilla AE", recon_vae.cpu()),
        ("Rekonstruksi Sparse AE", recon_sae.cpu()),
        ("Rekonstruksi VAE", recon_v.cpu())
    ]
    for r_idx, (title, batch) in enumerate(rows):
        for c_idx in range(8):
            axes[r_idx, c_idx].imshow(batch[c_idx, 0].numpy(), cmap="gray")
            axes[r_idx, c_idx].axis("off")
            if c_idx == 0:
                axes[r_idx, c_idx].set_title(title, loc="left", fontsize=10)
    plt.tight_layout()
    plt.savefig("./figures/Figur_1_Rekonstruksi_SideBySide.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figur 2: 4x4 generative sampling
    print("Menyimpan Figur 2...")
    with torch.no_grad():
        samp_vae = vae.decode(torch.randn(16, 64, device=device)).cpu()
        samp_vgan = v_netG(torch.randn(16, 64, device=device)).cpu()
        samp_dcgan = dc_netG(torch.randn(16, 100, device=device)).cpu()
        samp_ddpm = ddpm.sample((16, 1, 28, 28)).cpu()

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    panels = [
        (axes[0, 0], "Panel A: Sampel VAE", samp_vae),
        (axes[0, 1], "Panel B: Sampel Vanilla GAN", samp_vgan),
        (axes[1, 0], "Panel C: Sampel DCGAN", samp_dcgan),
        (axes[1, 1], "Panel D: Sampel DDPM", samp_ddpm),
    ]
    for ax_panel, panel_title, images in panels:
        grid_img = np.zeros((4 * 28, 4 * 28))
        for idx in range(16):
            r, c = idx // 4, idx % 4
            grid_img[r*28:(r+1)*28, c*28:(c+1)*28] = images[idx, 0].numpy()
        ax_panel.imshow(grid_img, cmap="gray")
        ax_panel.set_title(panel_title, fontsize=11)
        ax_panel.axis("off")
    plt.tight_layout()
    plt.savefig("./figures/Figur_2_Random_Generative_Sampling.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figur 4: Interpolasi laten
    print("Menyimpan Figur 4...")
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    batch_x, _ = next(iter(test_loader))
    x_A = batch_x[0:1].to(device)
    x_B = batch_x[1:2].to(device)

    with torch.no_grad():
        zA_ae = vanilla_ae.encode(x_A)
        zB_ae = vanilla_ae.encode(x_B)

        zA_sae = sparse_ae.encode(x_A)
        zB_sae = sparse_ae.encode(x_B)

        muA, _ = vae.encode(x_A)
        muB, _ = vae.encode(x_B)

        zA_vgan = torch.randn(1, 64, device=device)
        zB_vgan = torch.randn(1, 64, device=device)

        zA_dcgan = torch.randn(1, 100, device=device)
        zB_dcgan = torch.randn(1, 100, device=device)

    fig, axes = plt.subplots(5, 6, figsize=(12, 9))
    models_interp = [
        ("Vanilla AE", vanilla_ae.decode, zA_ae, zB_ae),
        ("Sparse AE", sparse_ae.decode, zA_sae, zB_sae),
        ("VAE", vae.decode, muA, muB),
        ("Vanilla GAN", v_netG, zA_vgan, zB_vgan),
        ("DCGAN", dc_netG, zA_dcgan, zB_dcgan)
    ]

    for row_idx, (m_name, dec_fn, zA, zB) in enumerate(models_interp):
        for col_idx, a in enumerate(alphas):
            z_interp = (1.0 - a) * zA + a * zB
            with torch.no_grad():
                img_interp = dec_fn(z_interp).cpu()[0, 0].numpy()
            axes[row_idx, col_idx].imshow(img_interp, cmap="gray")
            axes[row_idx, col_idx].axis("off")
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(f"α = {a}", fontsize=10)
            if col_idx == 0:
                axes[row_idx, col_idx].set_title(m_name, loc="left", fontsize=10)

    plt.tight_layout()
    plt.savefig("./figures/Figur_4_Latent_Interpolation.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figur 5: Loss Curves & Denoising Trajectory
    print("Menyimpan Figur 5...")
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    if "vanilla_ae" in history:
        ax1.plot(history["vanilla_ae"]["losses"], label="Vanilla AE")
    if "sparse_ae" in history:
        ax1.plot(history["sparse_ae"]["losses"], label="Sparse AE")
    if "vae" in history:
        ax1.plot(history["vae"]["losses"], label="VAE")
    ax1.set_title("Training Loss: Autoencoders & VAE", fontsize=10)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = fig.add_subplot(gs[1, 0])
    if "vanilla_gan" in history:
        ax2.plot(history["vanilla_gan"]["g_losses"], label="Vanilla GAN (G)", linestyle="--", alpha=0.7)
        ax2.plot(history["vanilla_gan"]["d_losses"], label="Vanilla GAN (D)", linestyle=":", alpha=0.7)
    if "dcgan" in history:
        ax2.plot(history["dcgan"]["g_losses"], label="DCGAN (G)")
        ax2.plot(history["dcgan"]["d_losses"], label="DCGAN (D)")
    if "ddpm" in history:
        ax2.plot(history["ddpm"]["losses"], label="DDPM", color="black", linewidth=1.5)
    ax2.set_title("Loss Curves: GAN & DDPM", fontsize=10)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax3 = fig.add_subplot(gs[:, 1])
    with torch.no_grad():
        all_steps, final_img = ddpm.sample((1, 1, 28, 28), return_all_timesteps=True)
        selected_timesteps = [0, 50, 100, 150, 200, 250, 299]
        
    diff_grid = np.zeros((28, len(selected_timesteps) * 28))
    for i, t_idx in enumerate(selected_timesteps):
        t_img = all_steps[0, t_idx, 0].cpu().numpy()
        diff_grid[:, i*28:(i+1)*28] = np.clip(t_img, 0, 1)

    ax3.imshow(diff_grid, cmap="gray")
    ax3.set_title("DDPM Denoising Trajectory (x_T -> x_0)", fontsize=11)
    ax3.axis("off")

    plt.tight_layout()
    plt.savefig("./figures/Figur_5_LossCurves_and_DiffusionTrajectory.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Evaluasi selesai. Hasil tersimpan di outputs/ dan figures/")

if __name__ == "__main__":
    run_evaluation_and_figures()
