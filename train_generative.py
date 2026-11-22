import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataset import get_dataloaders
from models import VanillaAE, VAE, VanillaGenerator, VanillaDiscriminator, DCGANGenerator, DCGANDiscriminator, SimpleUNet, DDPM

def train_all_models(device="cuda" if torch.cuda.is_available() else "cpu", epochs=15):
    print(f"Device: {device}")
    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128, seed=42)
    history = {}

    # 1. Vanilla Autoencoder
    print("Training Vanilla Autoencoder...")
    ae = VanillaAE(latent_dim=64).to(device)
    optimizer_ae = optim.Adam(ae.parameters(), lr=1e-3)
    criterion_mse = nn.MSELoss()
    ae_losses, ae_times = [], []

    for epoch in range(epochs):
        t0 = time.time()
        ae.train()
        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer_ae.zero_grad()
            recon, _ = ae(x)
            loss = criterion_mse(recon, x)
            loss.backward()
            optimizer_ae.step()
            total_loss += loss.item() * x.size(0)
        t_epoch = time.time() - t0
        avg_loss = total_loss / len(train_loader.dataset)
        ae_losses.append(avg_loss)
        ae_times.append(t_epoch)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Vanilla AE | Epoch {epoch+1:02d}/{epochs} - loss: {avg_loss:.5f} ({t_epoch:.1f}s)")
    
    torch.save(ae.state_dict(), "./checkpoints/vanilla_ae.pt")
    history["vanilla_ae"] = {"losses": ae_losses, "avg_time_epoch": sum(ae_times)/len(ae_times)}

    # 2. Sparse Autoencoder
    print("Training Sparse Autoencoder...")
    sparse_ae = VanillaAE(latent_dim=64).to(device)
    optimizer_sp = optim.Adam(sparse_ae.parameters(), lr=1e-3)
    sp_losses, sp_times = [], []
    lambda_l1 = 1e-4

    for epoch in range(epochs):
        t0 = time.time()
        sparse_ae.train()
        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer_sp.zero_grad()
            recon, z = sparse_ae(x)
            recon_loss = criterion_mse(recon, x)
            l1_loss = lambda_l1 * torch.mean(torch.abs(z))
            loss = recon_loss + l1_loss
            loss.backward()
            optimizer_sp.step()
            total_loss += loss.item() * x.size(0)
        t_epoch = time.time() - t0
        avg_loss = total_loss / len(train_loader.dataset)
        sp_losses.append(avg_loss)
        sp_times.append(t_epoch)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Sparse AE  | Epoch {epoch+1:02d}/{epochs} - loss: {avg_loss:.5f} ({t_epoch:.1f}s)")

    torch.save(sparse_ae.state_dict(), "./checkpoints/sparse_ae.pt")
    history["sparse_ae"] = {"losses": sp_losses, "avg_time_epoch": sum(sp_times)/len(sp_times)}

    # 3. Variational Autoencoder (VAE)
    print("Training VAE...")
    vae = VAE(latent_dim=64).to(device)
    optimizer_vae = optim.Adam(vae.parameters(), lr=1e-3)
    vae_losses, vae_times = [], []

    for epoch in range(epochs):
        t0 = time.time()
        vae.train()
        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer_vae.zero_grad()
            recon, mu, logvar, _ = vae(x)
            recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = (recon_loss + kl_loss) / x.size(0)
            loss.backward()
            optimizer_vae.step()
            total_loss += loss.item() * x.size(0)
        t_epoch = time.time() - t0
        avg_loss = total_loss / len(train_loader.dataset)
        vae_losses.append(avg_loss)
        vae_times.append(t_epoch)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"VAE        | Epoch {epoch+1:02d}/{epochs} - loss: {avg_loss:.5f} ({t_epoch:.1f}s)")

    torch.save(vae.state_dict(), "./checkpoints/vae.pt")
    history["vae"] = {"losses": vae_losses, "avg_time_epoch": sum(vae_times)/len(vae_times)}

    # 4. Vanilla GAN
    print("Training Vanilla GAN...")
    v_netG = VanillaGenerator(latent_dim=64).to(device)
    v_netD = VanillaDiscriminator().to(device)
    opt_vG = optim.Adam(v_netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_vD = optim.Adam(v_netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce_logits = nn.BCEWithLogitsLoss()
    v_g_losses, v_d_losses, v_times = [], [], []

    for epoch in range(epochs):
        t0 = time.time()
        v_netG.train()
        v_netD.train()
        g_loss_acc, d_loss_acc = 0.0, 0.0
        for x, _ in train_loader:
            b_size = x.size(0)
            x = x.to(device)
            real_labels = torch.ones(b_size, 1, device=device)
            fake_labels = torch.zeros(b_size, 1, device=device)

            opt_vD.zero_grad()
            out_real = v_netD(x)
            loss_d_real = bce_logits(out_real, real_labels)

            z = torch.randn(b_size, 64, device=device)
            fake_x = v_netG(z)
            out_fake = v_netD(fake_x.detach())
            loss_d_fake = bce_logits(out_fake, fake_labels)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_vD.step()

            opt_vG.zero_grad()
            out_gen = v_netD(fake_x)
            loss_g = bce_logits(out_gen, real_labels)
            loss_g.backward()
            opt_vG.step()

            g_loss_acc += loss_g.item() * b_size
            d_loss_acc += loss_d.item() * b_size

        t_epoch = time.time() - t0
        avg_g = g_loss_acc / len(train_loader.dataset)
        avg_d = d_loss_acc / len(train_loader.dataset)
        v_g_losses.append(avg_g)
        v_d_losses.append(avg_d)
        v_times.append(t_epoch)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Vanilla GAN| Epoch {epoch+1:02d}/{epochs} - D loss: {avg_d:.4f} - G loss: {avg_g:.4f}")

    torch.save(v_netG.state_dict(), "./checkpoints/vanilla_gan_g.pt")
    torch.save(v_netD.state_dict(), "./checkpoints/vanilla_gan_d.pt")
    history["vanilla_gan"] = {"g_losses": v_g_losses, "d_losses": v_d_losses, "avg_time_epoch": sum(v_times)/len(v_times)}

    # 5. DCGAN
    print("Training DCGAN...")
    dc_netG = DCGANGenerator(latent_dim=100).to(device)
    dc_netD = DCGANDiscriminator().to(device)
    opt_dcG = optim.Adam(dc_netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_dcD = optim.Adam(dc_netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
    dc_g_losses, dc_d_losses, dc_times = [], [], []

    for epoch in range(epochs):
        t0 = time.time()
        dc_netG.train()
        dc_netD.train()
        g_loss_acc, d_loss_acc = 0.0, 0.0
        for x, _ in train_loader:
            b_size = x.size(0)
            x = x.to(device)
            real_labels = torch.ones(b_size, 1, device=device)
            fake_labels = torch.zeros(b_size, 1, device=device)

            opt_dcD.zero_grad()
            out_real = dc_netD(x)
            loss_d_real = bce_logits(out_real, real_labels)

            z = torch.randn(b_size, 100, device=device)
            fake_x = dc_netG(z)
            out_fake = dc_netD(fake_x.detach())
            loss_d_fake = bce_logits(out_fake, fake_labels)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_dcD.step()

            opt_dcG.zero_grad()
            out_gen = dc_netD(fake_x)
            loss_g = bce_logits(out_gen, real_labels)
            loss_g.backward()
            opt_dcG.step()

            g_loss_acc += loss_g.item() * b_size
            d_loss_acc += loss_d.item() * b_size

        t_epoch = time.time() - t0
        avg_g = g_loss_acc / len(train_loader.dataset)
        avg_d = d_loss_acc / len(train_loader.dataset)
        dc_g_losses.append(avg_g)
        dc_d_losses.append(avg_d)
        dc_times.append(t_epoch)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"DCGAN      | Epoch {epoch+1:02d}/{epochs} - D loss: {avg_d:.4f} - G loss: {avg_g:.4f}")

    torch.save(dc_netG.state_dict(), "./checkpoints/dcgan_g.pt")
    torch.save(dc_netD.state_dict(), "./checkpoints/dcgan_d.pt")
    history["dcgan"] = {"g_losses": dc_g_losses, "d_losses": dc_d_losses, "avg_time_epoch": sum(dc_times)/len(dc_times)}

    # 6. DDPM
    print("Training DDPM...")
    unet = SimpleUNet(in_channels=1, base_channels=32).to(device)
    ddpm = DDPM(network=unet, timesteps=300, device=device)
    optimizer_ddpm = optim.Adam(unet.parameters(), lr=1e-3)
    ddpm_losses, ddpm_times = [], []

    for epoch in range(epochs):
        t0 = time.time()
        unet.train()
        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            b_size = x.size(0)
            t = torch.randint(0, ddpm.timesteps, (b_size,), device=device).long()
            
            optimizer_ddpm.zero_grad()
            loss = ddpm.p_losses(x, t)
            loss.backward()
            optimizer_ddpm.step()
            total_loss += loss.item() * b_size

        t_epoch = time.time() - t0
        avg_loss = total_loss / len(train_loader.dataset)
        ddpm_losses.append(avg_loss)
        ddpm_times.append(t_epoch)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"DDPM       | Epoch {epoch+1:02d}/{epochs} - loss: {avg_loss:.5f} ({t_epoch:.1f}s)")

    torch.save(unet.state_dict(), "./checkpoints/ddpm_unet.pt")
    history["ddpm"] = {"losses": ddpm_losses, "avg_time_epoch": sum(ddpm_times)/len(ddpm_times)}

    with open("./outputs/loss_history.json", "w") as f:
        json.dump(history, f, indent=4)
    print("Pelatihan selesai. Checkpoint tersimpan di checkpoints/")

if __name__ == "__main__":
    train_all_models()
