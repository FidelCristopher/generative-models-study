import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms
import numpy as np

def get_dataloaders(data_dir="./data", batch_size=128, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_raw = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_raw = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    # Menggabungkan seluruh 70.000 citra untuk pembagian 80-10-10 terstandarisasi
    full_dataset = ConcatDataset([train_raw, test_raw])
    total_size = len(full_dataset)
    train_size = int(0.80 * total_size)
    val_size = int(0.10 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
