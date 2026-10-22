"""CIFAR-10 ResNet-56 and VGG-16-BN with the public pretrained weights of
chenyaofo/pytorch-cifar-models (state-dict compatible definitions).

Used to (1) export the ONNX base models h3dnas compresses and (2) give the
source-code baselines (Torch-Pruning / DepGraph) the identical PyTorch model.
"""

from __future__ import annotations

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
}
PUBLISHED_TOP1 = {"resnet56": 94.37, "vgg16_bn": 94.16}


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


def load(name: str, weights_dir: Path) -> nn.Module:
    model = CifarResNet(56) if name == "resnet56" else CifarVGG()
    path = weights_dir / f"{name}.pt"
    if not path.exists():
        weights_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(WEIGHTS[name], str(path))
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.eval()


def export(model: nn.Module, path: Path) -> None:
    """ONNX opset 13 with BatchNormalization kept as separate nodes (as shipped models do)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model.eval(), torch.zeros(1, 3, 32, 32), str(path), opset_version=13,
                      input_names=["x"], output_names=["y"],
                      dynamic_axes={"x": {0: "n"}, "y": {0: "n"}},
                      training=torch.onnx.TrainingMode.PRESERVE, do_constant_folding=False,
                      dynamo=False)


if __name__ == "__main__":
    import argparse
    here = Path(__file__).resolve().parent                 # models/scripts
    ap = argparse.ArgumentParser(description="Export the CIFAR-10 base models (chenyaofo weights)")
    ap.add_argument("--model", choices=["resnet56", "vgg16_bn"], help="default: both")
    ap.add_argument("--out", type=Path, help="output file (with --model); default: its path in models/models.yaml")
    a = ap.parse_args()
    names = {"resnet56": "cifar_resnet56", "vgg16_bn": "cifar_vgg16"}
    for arch in ([a.model] if a.model else list(names)):
        if a.out and a.model:
            out = a.out
        else:
            from h3dnas.workspace import load as load_workspace
            out = load_workspace().onnx(names[arch])
        export(load(arch, here / "weights"), out)
        print(f"exported {arch} -> {out}")
