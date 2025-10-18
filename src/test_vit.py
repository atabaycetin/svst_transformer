import os
import glob
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import ViTConfig, ViTModel

from prepare_data_vit import ViTDataset
from prepare_data_ts_transformer import load_dataframe


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
CSV_FILE = "farm_data.csv"


def is_timeseries_checkpoint(state_dict: dict) -> bool:
    for k, v in state_dict.items():
        if "time_embeddings" in k and isinstance(v, torch.Tensor) and v.ndim == 3 and v.shape[0] == 1:
            return True
    return False


def is_vit_checkpoint(state_dict: dict) -> bool:
    for k, v in state_dict.items():
        if "vit.embeddings.patch_embeddings.projection.weight" in k:
            return True
    for k in state_dict.keys():
        if "vit.embeddings.patch_embeddings.projection" in k:
            return True
    return any(k.startswith("vit.") and ".embeddings." in k for k in state_dict.keys())


def infer_vit_params(state_dict: dict):
    """
    Infer num_channels, patch_size, hidden_size, num_hidden_layers, num_classes
    from common HF ViT checkpoint key shapes.
    """
    in_ch = 9
    patch = 16
    hidden = 768
    layers = 12
    num_classes = 2

    proj_key = None
    for k, v in state_dict.items():
        if k.endswith("vit.embeddings.patch_embeddings.projection.weight") and v.ndim == 4:
            proj_key = k
            break
    if proj_key is None:
        for k, v in state_dict.items():
            if "vit." in k and "projection.weight" in k and isinstance(v, torch.Tensor) and v.ndim == 4:
                proj_key = k
                break

    if proj_key is not None:
        W = state_dict[proj_key]
        hidden = int(W.shape[0])
        in_ch = int(W.shape[1])
        if W.shape[2] == W.shape[3]:
            patch = int(W.shape[2])

    layer_ids = set()
    for k in state_dict.keys():
        if "vit.encoder.layer." in k:
            try:
                idx = int(k.split("vit.encoder.layer.")[1].split(".")[0])
                layer_ids.add(idx)
            except Exception:
                pass
    if layer_ids:
        layers = max(layer_ids) + 1

    for key in ("classifier.weight", "vit.pooler.dense.weight"):
        if key in state_dict and isinstance(state_dict[key], torch.Tensor) and state_dict[key].ndim == 2:
            num_classes = int(state_dict[key].shape[0])
            break

    return in_ch, patch, hidden, layers, num_classes


class ViTForClassification(nn.Module):
    def __init__(self, num_channels=9, img_size=224, patch_size=16,
                 hidden_size=768, num_hidden_layers=12, num_classes=2):
        super().__init__()
        config = ViTConfig(
            image_size=img_size,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
        )
        self.vit = ViTModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(self, x):  # x: [B, 9, 224, 224]
        out = self.vit(pixel_values=x)       # last_hidden_state [B, tokens, H]
        pooled = out.last_hidden_state[:, 0] # CLS token
        return self.classifier(pooled)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    total_conf = 0.0
    per_class = Counter()
    per_class_correct = Counter()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.long().to(device)

            logits = model(images)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

            correct += (pred == labels).sum().item()
            total += labels.size(0)
            total_conf += conf.sum().item()

            for y, p in zip(labels.tolist(), pred.tolist()):
                per_class[y] += 1
                if y == p:
                    per_class_correct[y] += 1

    acc = correct / max(1, total)
    mean_conf = total_conf / max(1, total)
    per_class_acc = {c: (per_class_correct[c] / per_class[c]) if per_class[c] > 0 else 0.0
                     for c in sorted(per_class.keys())}
    return acc, mean_conf, per_class_acc, total


def build_loader(csv_path, batch_size=8, num_workers=2):
    dataset = ViTDataset(csv_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt_paths = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth")))
    if not ckpt_paths:
        print(f"[error] No .pth files found in: {os.path.abspath(CHECKPOINT_DIR)}")
        return
    print(f"Found {len(ckpt_paths)} checkpoint(s) in {CHECKPOINT_DIR}.")

    if not os.path.exists(os.path.join(ROOT, CSV_FILE)):
        print(f"[error] Missing CSV: {CSV_FILE}")
        return
    df = load_dataframe(CSV_FILE)
    n_fields = df["Field_ID"].nunique()
    print(f"CSV: {CSV_FILE} | fields={n_fields}")

    results = []

    for ckpt_path in ckpt_paths:
        fname = os.path.basename(ckpt_path)
        print("\n" + "=" * 80)
        print(f"Evaluating checkpoint: {fname}")

        obj = torch.load(ckpt_path, map_location="cpu")
        state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj

        if is_timeseries_checkpoint(state):
            print(f"[skip] {fname} looks like a time-series transformer (has time_embeddings).")
            continue

        if not is_vit_checkpoint(state):
            print(f"[skip] {fname} does not look like a ViT checkpoint.")
            continue

        in_ch, patch, hidden, layers, ncls = infer_vit_params(state)
        print(f"Inferred ViT params: in_ch={in_ch}, patch={patch}, hidden={hidden}, layers={layers}, classes={ncls}")

        model = ViTForClassification(
            num_channels=in_ch,
            img_size=224,
            patch_size=patch,
            hidden_size=hidden,
            num_hidden_layers=layers,
            num_classes=ncls
        ).to(device)

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:    print(f"[warn] Missing keys: {missing}")
        if unexpected: print(f"[warn] Unexpected keys: {unexpected}")

        loader = build_loader(os.path.join(ROOT, CSV_FILE), batch_size=8, num_workers=2)

        acc, mean_conf, per_class_acc, n = evaluate(model, loader, device)
        print(f"Samples: {n} | Accuracy: {acc:.4f} | Mean confidence: {mean_conf:.4f}")
        for c, a in per_class_acc.items():
            print(f"  Class {c} accuracy: {a:.4f}")

        results.append((fname, acc, mean_conf))

    print("\n" + "#" * 80)
    print("SUMMARY (ViT checkpoints only):")
    if not results:
        print("No ViT checkpoints evaluated.")
        return
    results_sorted = sorted(results, key=lambda x: (x[1], x[2]), reverse=True)
    for fname, acc, mean_conf in results_sorted:
        print(f"{fname:40s} | Acc={acc:.4f} | MeanConf={mean_conf:.4f}")


if __name__ == "__main__":
    main()
