import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# MLP Classifier & MLP Decoder untuk pembuktian batasan teoritis
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=64, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)    # W1: 256 x 784
        self.fc2 = nn.Linear(hidden_dim, latent_dim)   # W2: 64 x 256
        self.fc3 = nn.Linear(latent_dim, num_classes)  # W3: 10 x 64

    def forward_features(self, x):
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x))
        z = F.relu(self.fc2(h1))
        return z

    def forward(self, x):
        z = self.forward_features(x)
        logits = self.fc3(z)
        return logits

class MLPDecoder(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=256, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)

# Vanilla Autoencoder & Sparse Autoencoder
class VanillaAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = x.view(x.size(0), -1)
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

# Variational Autoencoder (VAE)
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

# Vanilla GAN
class VanillaGenerator(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)

class VanillaDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

# Deep Convolutional GAN (DCGAN)
class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 128, 7, 7)
        return self.deconv(h)

class DCGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1)
        )

    def forward(self, x):
        return self.net(x)

# DDPM Components
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, time_emb_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        
        self.mid_conv = nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)
        self.out_conv = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.down1(h1))

        h_mid = F.relu(self.mid_conv(h2) + self.time_proj(t_emb)[:, :, None, None])

        h_up = F.relu(self.up1(h_mid))
        h_cat = torch.cat([h_up, h1], dim=1)
        h_out = F.relu(self.conv2(h_cat))
        return self.out_conv(h_out)

class DDPM(nn.Module):
    def __init__(self, network, timesteps=500, beta_start=1e-4, beta_end=0.02, device="cpu"):
        super().__init__()
        self.network = network
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def p_losses(self, x_start, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.network(x_noisy, t)
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x, t):
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = (1.0 / torch.sqrt(self.alphas[t])).view(-1, 1, 1, 1)

        model_mean = sqrt_recip_alphas_t * (x - (betas_t / sqrt_one_minus_alphas_cumprod_t) * self.network(x, t))

        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = (self.betas[t] * (1.0 - (self.alphas_cumprod[t - 1] if t[0] > 0 else 0.0)) / (1.0 - self.alphas_cumprod[t])).view(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, shape, return_all_timesteps=False):
        b = shape[0]
        img = torch.randn(shape, device=self.device)
        imgs = [img]

        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t)
            if return_all_timesteps:
                imgs.append(img)

        img = torch.clamp(img, 0.0, 1.0)
        return (torch.stack(imgs, dim=1), img) if return_all_timesteps else img
