"""
Train a simple Fashion MNIST classifier and export to ONNX.

    python models/scripts/fashion_mnist.py --data data/fashion_mnist --out models/base/fashion_mnist.onnx

Retraining gives a different file from the recorded one (see sources.yaml).
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import sys
import numpy as np

SCRIPTS = Path(__file__).resolve().parent             # models/scripts
MODELS = SCRIPTS.parent                               # models
ROOT = MODELS.parent    
_DATA = ROOT / "data/fashion_mnist"       # set by --data

# The h3dnas package (fetch.py also puts it on PYTHONPATH)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submodules" / "h3dnas"))

from h3dnas.utils.logger import get_logger

logger = get_logger(__name__)


# ===== 1. Define a tiny CNN =====
class FashionMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14x14
            nn.Conv2d(16, 32, 3, padding=1),  # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 7x7
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


def get_test_loader(batch_size=64):
    """Get test data loader for evaluation."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = datasets.FashionMNIST(
        root=str(_DATA),
        train=False,
        download=True,
        transform=transform
    )

    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def train_model(epochs=3, batch_size=64, return_loader=False):
    """
    Train the Fashion MNIST model.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        return_loader: If True, return (model, test_loader), else just model

    Returns:
        model or (model, test_loader)
    """
    logger.info("=" * 70)
    logger.info("Training Fashion MNIST Classifier")
    logger.info("=" * 70)

    # Seed everything so the exported fashion_mnist.onnx is identical
    # regardless of machine or OS. This must match across all collaborators.
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load dataset
    logger.info("Loading Fashion MNIST dataset...")
    train_dataset = datasets.FashionMNIST(
        root=str(_DATA),
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root=str(_DATA),
        train=False,
        download=True,
        transform=transform
    )

    g = torch.Generator()
    g.manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              generator=g, worker_init_fn=lambda _: np.random.seed(42))
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # Create model
    model = FashionMNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Training loop
    logger.info(f"\nStarting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 200 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | "
                           f"Loss: {loss.item():.4f}")

        train_acc = 100. * correct / total
        logger.info(f"Epoch {epoch+1}/{epochs} completed | "
                   f"Loss: {train_loss/len(train_loader):.4f} | "
                   f"Accuracy: {train_acc:.2f}%")

    # Test
    logger.info("\nEvaluating on test set...")
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_acc = 100. * correct / total
    logger.info(f"Test Loss: {test_loss/len(test_loader):.4f}")
    logger.info(f"Test Accuracy: {test_acc:.2f}%")

    if return_loader:
        return model, test_loader
    return model


def export_to_onnx(model, output_path="fashion_mnist.onnx"):
    """Export PyTorch model to ONNX format."""
    logger.info("\n" + "=" * 70)
    logger.info("Exporting to ONNX")
    logger.info("=" * 70)

    # Move model to CPU for export
    model = model.cpu()
    model.eval()

    # Create dummy input (batch_size=1, channels=1, height=28, width=28)
    dummy_input = torch.randn(1, 1, 28, 28)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export
    logger.info(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=False,   # keep Reshape shapes as initializers so NAS pruner can update them
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        dynamo=False,
    )

    logger.info(f" ONNX model saved to {output_path}")
    logger.info(f"   File size: {output_path.stat().st_size / 1024:.2f} KB")

    return output_path


def main():
    global _DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=False, default=_DATA, help="Fashion-MNIST folder (downloaded if empty)")
    ap.add_argument("--out", type=Path, required=False, default=MODELS / "base/fashion_mnist.onnx")
    a = ap.parse_args()
    _DATA = a.data
    model = train_model(epochs=10, batch_size=64)
    export_to_onnx(model, a.out)
    logger.info(f"Saved {a.out}")


if __name__ == "__main__":
    main()
