import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from dataset import get_dataloaders
from models import MLPClassifier, MLPDecoder

def run_mlp_experiments(device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Device: {device}")
    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./figures", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128, seed=42)

    # 1. Pelatihan MLP Classifier
    print("Training MLP classifier...")
    mlp = MLPClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=1e-3)

    num_epochs = 15
    for epoch in range(num_epochs):
        mlp.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = mlp(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        mlp.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = mlp(x).argmax(dim=-1)
                val_correct += (preds == y).sum().item()
                val_total += x.size(0)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{num_epochs:02d} - Train Acc: {correct/total*100:.2f}% - Val Acc: {val_correct/val_total*100:.2f}%")

    test_correct, test_total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = mlp(x).argmax(dim=-1)
            test_correct += (preds == y).sum().item()
            test_total += x.size(0)
    test_acc = test_correct / test_total * 100
    print(f"Test Accuracy: {test_acc:.2f}%")
    torch.save(mlp.state_dict(), "./checkpoints/mlp_classifier.pt")

    W1 = mlp.fc1.weight.detach()
    W2 = mlp.fc2.weight.detach()
    W3 = mlp.fc3.weight.detach()

    sample_images, sample_labels = next(iter(test_loader))
    sample_images = sample_images.to(device)
    
    with torch.no_grad():
        z_mlp_sample = mlp.forward_features(sample_images)

    # 2. Proyeksi balik W^T dari vektor kelas target
    recon_classes = []
    for c in range(10):
        y_c = torch.zeros(10, device=device)
        y_c[c] = 1.0
        z_hat = torch.relu(torch.matmul(W3.t(), y_c))
        h1_hat = torch.relu(torch.matmul(W2.t(), z_hat))
        x_hat = torch.sigmoid(torch.matmul(W1.t(), h1_hat)).view(1, 28, 28)
        recon_classes.append(x_hat.cpu())
    recon_classes = torch.stack(recon_classes)

    # 3. Proyeksi balik dari fitur laten citra uji
    h1_from_z = torch.relu(torch.matmul(z_mlp_sample, W2))
    x_hat_from_z = torch.sigmoid(torch.matmul(h1_from_z, W1)).view(-1, 1, 28, 28)

    # 4. Frozen Latent Decoder
    print("Melatih decoder dari fitur laten MLP beku...")
    for param in mlp.parameters():
        param.requires_grad = False
    
    frozen_decoder = MLPDecoder(latent_dim=64).to(device)
    optimizer_dec = optim.Adam(frozen_decoder.parameters(), lr=1e-3)
    mse_criterion = nn.MSELoss()

    for epoch in range(10):
        frozen_decoder.train()
        epoch_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            with torch.no_grad():
                z = mlp.forward_features(x)
            optimizer_dec.zero_grad()
            recon = frozen_decoder(z)
            loss = mse_criterion(recon, x)
            loss.backward()
            optimizer_dec.step()
            epoch_loss += loss.item() * x.size(0)
        if (epoch + 1) % 5 == 0:
            print(f"Decoder Epoch {epoch+1:02d}/10 - Loss: {epoch_loss/len(train_loader.dataset):.5f}")

    torch.save(frozen_decoder.state_dict(), "./checkpoints/frozen_decoder.pt")

    frozen_decoder.eval()
    with torch.no_grad():
        recon_frozen = frozen_decoder(z_mlp_sample).cpu()

    # 5. Sampling dari prior ruang laten
    all_z = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            all_z.append(mlp.forward_features(x))
    all_z = torch.cat(all_z, dim=0)
    mu_z = all_z.mean(dim=0)
    sigma_z = all_z.std(dim=0)

    z_rand = torch.randn(16, 64, device=device) * sigma_z + mu_z
    with torch.no_grad():
        samples_from_mlp = frozen_decoder(z_rand).cpu()

    # Menyimpan visualisasi
    fig, axes = plt.subplots(4, 8, figsize=(14, 8))
    for i in range(8):
        axes[0, i].imshow(sample_images[i, 0].cpu().numpy(), cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Ground Truth", loc="left", fontsize=10)

        axes[1, i].imshow(x_hat_from_z[i, 0].cpu().numpy(), cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Transpose Proj W^T", loc="left", fontsize=10)

        axes[2, i].imshow(recon_frozen[i, 0].numpy(), cmap="gray")
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_title("Frozen Latent Decoder", loc="left", fontsize=10)

        axes[3, i].imshow(samples_from_mlp[i, 0].numpy(), cmap="gray")
        axes[3, i].axis("off")
        if i == 0:
            axes[3, i].set_title("MLP Prior Sampling", loc="left", fontsize=10)

    plt.tight_layout()
    plt.savefig("./figures/Figur_3_Batasan_MLP.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig_c, axes_c = plt.subplots(1, 10, figsize=(15, 2))
    class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    for c in range(10):
        axes_c[c].imshow(recon_classes[c, 0].numpy(), cmap="gray")
        axes_c[c].set_title(class_names[c], fontsize=9)
        axes_c[c].axis("off")
    plt.tight_layout()
    plt.savefig("./figures/Figur_3_TopDown_Classes.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Figur batasan MLP tersimpan di figures/")

if __name__ == "__main__":
    run_mlp_experiments()
