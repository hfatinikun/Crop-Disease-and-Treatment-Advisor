"""
classifier.py
=============
Model definition for the tomato disease classifier.

Design
------
We use transfer learning from a pretrained ImageNet backbone rather than
training a CNN from scratch. This is the correct choice here because:
  - PlantVillage has ~18k tomato images after filtering — too few to train
    a deep CNN from random weights without severe overfitting.
  - ImageNet features (edges, textures, colour blobs) transfer well to
    plant disease patterns (lesion shapes, discolouration, texture changes).

Two backbone options are provided:
  - EfficientNet-B2  (default) : best accuracy/parameter trade-off for this
                                  image size and dataset scale. ~9M params.
  - ResNet-50                  : well-understood baseline, easier to debug.
                                  ~25M params.

Fine-tuning strategy — three phases (handled in train.py):
  Phase 1 — Feature extraction  : backbone frozen, only classifier head trained.
                                   Run for ~5 epochs. Fast, low LR.
  Phase 2 — Fine-tuning         : unfreeze last N backbone layers, train end-to-end.
                                   Lower LR for backbone, higher for head.
  Phase 3 — (optional) Full ft  : unfreeze everything at very low LR.

Usage (standalone — prints model summary)
-----------------------------------------
  python classifier.py --backbone efficientnet --num-classes 10
  python classifier.py --backbone resnet50     --num-classes 10 --summary

Usage (from train.py)
---------------------
  from classifier import build_model, freeze_backbone, unfreeze_last_n_layers

  model = build_model(backbone="efficientnet", num_classes=10, pretrained=True)
  freeze_backbone(model)           # Phase 1
  # ... train head only ...
  unfreeze_last_n_layers(model, n=3)   # Phase 2
  # ... fine-tune ...
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models

log = logging.getLogger(__name__)

# ── supported backbones ────────────────────────────────────────────────────────

SUPPORTED_BACKBONES = ["efficientnet", "resnet50"]


# ── model builder ──────────────────────────────────────────────────────────────

def build_model(
    backbone:    str  = "efficientnet",
    num_classes: int  = 10,
    pretrained:  bool = True,
    dropout:     float = 0.4,
) -> nn.Module:
    """
    Build and return a classification model with a custom head.

    Parameters
    ----------
    backbone    : "efficientnet" | "resnet50"
    num_classes : number of output classes (10 for tomato diseases)
    pretrained  : load ImageNet weights (always True unless ablation study)
    dropout     : dropout rate in the classification head

    Returns
    -------
    nn.Module with .backbone and .classifier attributes for easy layer access.
    """
    backbone = backbone.lower()
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unknown backbone {backbone!r}. Choose from {SUPPORTED_BACKBONES}.")

    if backbone == "efficientnet":
        return _build_efficientnet(num_classes, pretrained, dropout)
    elif backbone == "resnet50":
        return _build_resnet50(num_classes, pretrained, dropout)


# ── EfficientNet-B2 ────────────────────────────────────────────────────────────

def _build_efficientnet(
    num_classes: int,
    pretrained:  bool,
    dropout:     float,
) -> nn.Module:
    """
    EfficientNet-B2 with a custom classification head.

    Architecture of our head (replacing the default single Linear layer):
        AdaptiveAvgPool → Flatten → Dropout → Linear(1408, 512)
        → BatchNorm → ReLU → Dropout → Linear(512, num_classes)

    The extra hidden layer + BN gives the head enough capacity to adapt
    ImageNet features to plant disease patterns without overfitting,
    especially in Phase 1 when the backbone is frozen.
    """
    weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
    base    = models.efficientnet_b2(weights=weights)

    in_features = base.classifier[1].in_features   # 1408 for B2

    # Replace the default head
    base.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout / 2),
        nn.Linear(512, num_classes),
    )

    # Name the parts for easy access in freeze/unfreeze helpers
    model = _NamedModel(
        backbone   = base.features,   # the convolutional feature extractor
        pooling    = base.avgpool,
        classifier = base.classifier,
        name       = "EfficientNet-B2",
    )
    return model


# ── ResNet-50 ──────────────────────────────────────────────────────────────────

def _build_resnet50(
    num_classes: int,
    pretrained:  bool,
    dropout:     float,
) -> nn.Module:
    """
    ResNet-50 with a custom classification head.

    Architecture of our head:
        AvgPool (built-in) → Flatten → Dropout → Linear(2048, 512)
        → BatchNorm → ReLU → Dropout → Linear(512, num_classes)
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    base    = models.resnet50(weights=weights)

    in_features = base.fc.in_features   # 2048 for ResNet-50

    # Replace the final fully-connected layer with our head
    base.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout / 2),
        nn.Linear(512, num_classes),
    )

    # ResNet's backbone = everything except the final fc
    backbone_layers = nn.Sequential(
        base.conv1, base.bn1, base.relu, base.maxpool,
        base.layer1, base.layer2, base.layer3, base.layer4,
    )

    model = _NamedModel(
        backbone   = backbone_layers,
        pooling    = nn.Sequential(base.avgpool, nn.Flatten()),
        classifier = base.fc,
        name       = "ResNet-50",
    )
    return model


# ── named model container ──────────────────────────────────────────────────────

class _NamedModel(nn.Module):
    """
    Thin wrapper that exposes .backbone, .pooling, and .classifier
    as named submodules regardless of which backbone is used.
    This makes the freeze/unfreeze helpers backbone-agnostic.
    """

    def __init__(
        self,
        backbone:   nn.Module,
        pooling:    nn.Module,
        classifier: nn.Module,
        name:       str = "",
    ):
        super().__init__()
        self.backbone   = backbone
        self.pooling    = pooling
        self.classifier = classifier
        self.model_name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pooling(x)
        if x.dim() > 2:
            x = x.flatten(1)
        x = self.classifier(x)
        return x

    def __repr__(self) -> str:
        total  = sum(p.numel() for p in self.parameters())
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return (
            f"{self.model_name}(\n"
            f"  total params  : {total:,}\n"
            f"  frozen params : {frozen:,}  ({100*frozen/total:.1f}%)\n"
            f"  trainable     : {total-frozen:,}  ({100*(total-frozen)/total:.1f}%)\n"
            f")"
        )


# ── freeze / unfreeze helpers ──────────────────────────────────────────────────

def freeze_backbone(model: _NamedModel) -> None:
    """
    Phase 1 — freeze all backbone parameters.
    Only the classifier head will be updated.
    Call this right after build_model() for the first training phase.
    """
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.pooling.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(
        "Backbone frozen. Trainable: %s / %s params (%.1f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )


def unfreeze_last_n_layers(model: _NamedModel, n: int = 3) -> None:
    """
    Phase 2 — unfreeze the last N children of the backbone for fine-tuning.
    Earlier layers keep their ImageNet weights frozen (they learn generic
    low-level features that don't need updating for leaf disease).

    For EfficientNet-B2: backbone has 9 children (blocks 0-8).
      n=3  unfreezes blocks 6, 7, 8  (last three MBConv groups)
    For ResNet-50: backbone has 8 children (conv1…layer4).
      n=3  unfreezes layer2, layer3, layer4

    Parameters
    ----------
    n : number of backbone children to unfreeze, counting from the end.
        Recommended starting values:
          n=2  conservative — only last two blocks
          n=3  default — good balance
          n=5  aggressive — use only if val loss has plateaued
    """
    # First unfreeze everything, then re-freeze selectively
    for param in model.parameters():
        param.requires_grad = True

    children = list(model.backbone.children())
    freeze_up_to = len(children) - n

    for i, child in enumerate(children):
        if i < freeze_up_to:
            for param in child.parameters():
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(
        "Unfroze last %d backbone layers. Trainable: %s / %s params (%.1f%%)",
        n, f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )


def unfreeze_all(model: _NamedModel) -> None:
    """Phase 3 — unfreeze the entire network for full fine-tuning at very low LR."""
    for param in model.parameters():
        param.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    log.info("All layers unfrozen. Trainable: %s params.", f"{total:,}")


# ── save / load ────────────────────────────────────────────────────────────────

def save_checkpoint(
    model:       _NamedModel,
    optimizer:   torch.optim.Optimizer,
    epoch:       int,
    val_acc:     float,
    class_names: list[str],
    path:        str | Path,
) -> None:
    """
    Save a full checkpoint: model weights, optimiser state, metadata.
    Always saves the best model (called from train.py when val_acc improves).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":       epoch,
        "val_acc":     val_acc,
        "model_name":  model.model_name,
        "class_names": class_names,
        "num_classes": len(class_names),
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
    }, path)
    log.info("Checkpoint saved → %s  (epoch %d, val_acc=%.4f)", path, epoch, val_acc)


def load_checkpoint(
    path:        str | Path,
    backbone:    str  = "efficientnet",
    num_classes: int  = 10,
    device:      str  = "cpu",
) -> tuple[_NamedModel, dict]:
    """
    Load a checkpoint and return (model, metadata_dict).
    The model is returned in eval() mode, ready for inference.

    Parameters
    ----------
    path        : checkpoint file path
    backbone    : must match what was used when saving
    num_classes : must match what was used when saving
    device      : "cpu" | "cuda" | "mps"
    """
    ckpt  = torch.load(path, map_location=device)
    model = build_model(backbone=backbone, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    log.info(
        "Loaded checkpoint from %s (epoch %d, val_acc=%.4f)",
        path, ckpt["epoch"], ckpt["val_acc"],
    )
    return model, ckpt


# ── standalone summary ─────────────────────────────────────────────────────────

def _count_params(model: nn.Module) -> tuple[int, int]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_layer_summary(model: _NamedModel) -> None:
    """Print a readable layer-by-layer summary of the backbone."""
    print(f"\n{'─'*60}")
    print(f"  {model.model_name}  —  layer summary")
    print(f"{'─'*60}")
    for i, (name, child) in enumerate(model.backbone.named_children()):
        n_params = sum(p.numel() for p in child.parameters())
        frozen   = all(not p.requires_grad for p in child.parameters())
        status   = "frozen" if frozen else "trainable"
        print(f"  [{i:2d}] backbone.{name:<25}  {n_params:>10,} params  [{status}]")
    print(f"{'─'*60}")
    total, trainable = _count_params(model)
    print(f"  Total params     : {total:>12,}")
    print(f"  Trainable params : {trainable:>12,}  ({100*trainable/total:.1f}%)")
    print(f"  Frozen params    : {total-trainable:>12,}  ({100*(total-trainable)/total:.1f}%)")
    print(f"{'─'*60}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build and inspect the tomato disease classifier.")
    p.add_argument("--backbone",     choices=SUPPORTED_BACKBONES, default="efficientnet")
    p.add_argument("--num-classes",  type=int,   default=10)
    p.add_argument("--dropout",      type=float, default=0.4)
    p.add_argument("--no-pretrained",action="store_true")
    p.add_argument("--summary",      action="store_true",
                   help="Print layer-by-layer summary after building the model.")
    p.add_argument("--test-forward", action="store_true",
                   help="Run a dummy forward pass to confirm shapes are correct.")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
                        datefmt="%H:%M:%S")

    args = parse_args()

    print(f"\nBuilding {args.backbone} (pretrained={not args.no_pretrained}) "
          f"for {args.num_classes} classes ...")

    model = build_model(
        backbone    = args.backbone,
        num_classes = args.num_classes,
        pretrained  = not args.no_pretrained,
        dropout     = args.dropout,
    )

    # Show param counts before / after freeze
    print("\n── All layers trainable ──")
    print(model)

    print("\n── After freeze_backbone() (Phase 1) ──")
    freeze_backbone(model)
    print(model)

    print("\n── After unfreeze_last_n_layers(n=3) (Phase 2) ──")
    unfreeze_last_n_layers(model, n=3)
    print(model)

    if args.summary:
        print_layer_summary(model)

    if args.test_forward:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        dummy = torch.randn(4, 3, 224, 224, device=device)
        with torch.no_grad():
            out = model(dummy)
        print(f"\nForward pass OK.")
        print(f"  Input : {tuple(dummy.shape)}")
        print(f"  Output: {tuple(out.shape)}  (batch_size × num_classes)")
        print(f"  Device: {device}")
