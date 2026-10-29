"""CIFAR-10 base models with public pretrained weights (state-dict compatible definitions):
ResNet-56 and VGG-16-BN of chenyaofo/pytorch-cifar-models, and ResNet-18 of
edadaltocg/resnet18_cifar10 (Hugging Face; the ONNXPruner comparison): a timm ResNet-18
with the CIFAR stem (3x3 stride-1 first conv, no max-pool) whose layer names match
torchvision's resnet18; inputs normalised with mean (0.4914, 0.4822, 0.4465) and std
(0.2023, 0.1994, 0.2010) as in its config.json (the same as chenyaofo's); model card
94.98% test accuracy (best epoch; last epoch 94.92%).

Used to (1) export the ONNX base models h3dnas compresses and (2) give the
source-code baselines (Torch-Pruning / DepGraph) the identical PyTorch model.

    python models/scripts/cifar_models.py   # downloads weights, verifies, exports
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import sys
import torch
import torch.nn as nn

SCRIPTS = Path(__file__).resolve().parent             # models/scripts
MODELS = SCRIPTS.parent                               # models
ROOT = MODELS.parent
H3DNAS = ROOT / "submodules" / "h3dnas"    
DATA = ROOT / "data/cifar10"       # set by --data
sys.path.insert(0, str(H3DNAS))

WEIGHTS = {
    "resnet56": "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt",
    "vgg16_bn": "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg16_bn-6ee7ea24.pt",
    "resnet18": "https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/"
                "ccacbae365101b9bfa45e1f17f4acac30c2601b2/pytorch_model.bin",
}
WEIGHTS_SHA256 = {"resnet18": "f656242526b567dbd46857276acbcfcb4ee74a7f8a7ae5e56557b7e8fa5d914a"}
WEIGHT_FILES = {"resnet18": "edadaltocg_resnet18_cifar10.bin"}    # default: <name>.pt
IO_NAMES = {"resnet18": ("image", "logits", "batch")}             # default: x, y, n
PUBLISHED_TOP1 = {"resnet56": 94.37, "vgg16_bn": 94.16, "resnet18": 94.98}
PUBLISHED_SOURCE = {"resnet56": "chenyaofo", "vgg16_bn": "chenyaofo", "resnet18": "edadaltocg model card"}
# Input normalisation (mean, std) each checkpoint was trained with.
NORMALIZE = {"resnet56": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
             "vgg16_bn": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
             "resnet18": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))}   # its config.json


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = None
        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                            nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + idt)


class CifarResNet(nn.Module):
    def __init__(self, depth=56, num_classes=10):
        super().__init__()
        n = (depth - 2) // 6
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        layers, cin = [], 16
        for i, c in enumerate((16, 32, 64)):
            blocks = []
            for j in range(n):
                blocks.append(BasicBlock(cin, c, 2 if (j == 0 and i > 0) else 1))
                cin = c
            layers.append(nn.Sequential(*blocks))
        self.layer1, self.layer2, self.layer3 = layers
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.fc(torch.flatten(self.avgpool(x), 1))


class CifarVGG(nn.Module):
    CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]

    def __init__(self, num_classes=10):
        super().__init__()
        layers, cin = [], 3
        for v in self.CFG:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [nn.Conv2d(cin, v, 3, padding=1), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                cin = v
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(),
            nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(),
            nn.Linear(512, num_classes))

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


def cifar_resnet18(num_classes=10) -> nn.Module:
    """torchvision ResNet-18 with the CIFAR stem: 3x3 stride-1 first conv, no max-pool."""
    import torchvision
    m = torchvision.models.resnet18(num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def load(name: str, weights_dir: Path) -> nn.Module:
    model = {"resnet56": lambda: CifarResNet(56), "vgg16_bn": CifarVGG,
             "resnet18": cifar_resnet18}[name]()
    path = weights_dir / WEIGHT_FILES.get(name, f"{name}.pt")
    want = WEIGHTS_SHA256.get(name)
    if not path.exists():
        weights_dir.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        torch.hub.download_url_to_file(WEIGHTS[name], str(part))
        if want and hashlib.sha256(part.read_bytes()).hexdigest() != want:
            part.unlink()
            raise ValueError(f"{WEIGHTS[name]}: downloaded file does not match the pinned sha256 "
                             f"(incomplete download?); run again")
        part.replace(path)
    if want and hashlib.sha256(path.read_bytes()).hexdigest() != want:
        raise ValueError(f"{path}: sha256 does not match the pinned {name} weights")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.eval()


def export(model: nn.Module, path: Path, names=("x", "y", "n")) -> None:
    """ONNX opset 13 with BatchNormalization kept as separate nodes (as shipped models do);
    names = (input, output, batch axis)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    i, o, b = names
    torch.onnx.export(model.eval(), torch.zeros(1, 3, 32, 32), str(path), opset_version=13,
                      input_names=[i], output_names=[o], dynamic_axes={i: {0: b}, o: {0: b}},
                      training=torch.onnx.TrainingMode.PRESERVE, do_constant_folding=False,
                      dynamo=False)


if __name__ == "__main__":
    import argparse
    here = Path(__file__).resolve().parent                 # models/scripts
    ap = argparse.ArgumentParser(description="Export the CIFAR-10 base models (pretrained weights)")
    ap.add_argument("--model", choices=list(WEIGHTS), help="default: all")
    ap.add_argument("--out", type=Path, help="output file (with --model); default: its path in models/models.yaml")
    a = ap.parse_args()
    names = {"resnet56": "cifar_resnet56", "vgg16_bn": "cifar_vgg16", "resnet18": "cifar_resnet18"}
    for arch in ([a.model] if a.model else list(names)):
        if a.out and a.model:
            out = a.out
        else:
            from h3dnas.workspace import load as load_workspace
            out = load_workspace().onnx(names[arch])
        export(load(arch, here / "weights"), out, IO_NAMES.get(arch, ("x", "y", "n")))
        print(f"exported {arch} -> {out}")
