import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import Counter

from prepare_data_ts_transformer import TimeSeriesImageDataset, load_dataframe
from train_ts_transformer import TimeSeriesTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
checkpoint_dir = os.path.join(ROOT, "checkpoints")
csv_path = "farm_data.csv"
DEFAULT_FALLBACK_T = 8

def is_timeseries_checkpoint(state_dict: dict) -> bool:
    """Return True if checkpoint looks like a time-series transformer (has time embeddings)."""
    for k, v in state_dict.items():
        if "time_embeddings" in k and isinstance(v, torch.Tensor) and v.ndim == 3 and v.shape[0] == 1:
            return True
    return False


def infer_num_frames_from_state(state_dict: dict, fallback: int) -> int:
    """Infer T from a 'time_embeddings' tensor of shape [1, T, dim]."""
    for k, v in state_dict.items():
        if "time_embeddings" in k and isinstance(v, torch.Tensor) and v.ndim == 3 and v.shape[0] == 1:
            return int(v.shape[1])
    return fallback


def maybe_interpolate_time_embeddings(model: TimeSeriesTransformer, state: dict) -> dict:
    """If checkpoint time embedding length != model.num_frames, interpolate along time."""
    key = None
    for k in state:
        if "time_embeddings" in k and isinstance(state[k], torch.Tensor) and state[k].ndim == 3:
            key = k
            break
    if key is None:
        return state

    ckpt_T = state[key].shape[1]
    model_T = model.num_frames
    if ckpt_T == model_T:
        return state

    with torch.no_grad():
        w = state[key]  # [1, T, dim]
        w_ch_first = w.permute(0, 2, 1)  # [1, dim, T]
        w_interp = F.interpolate(w_ch_first, size=model_T, mode="linear", align_corners=False)
        state[key] = w_interp.permute(0, 2, 1).contiguous()  # [1, model_T, dim]
    return state


def evaluate(model, loader, device, verbose=False):
    model.eval()
    correct = 0
    total = 0
    total_conf = 0.0
    per_class = Counter()
    per_class_correct = Counter()

    with torch.no_grad():
        sample_idx = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.long().to(device)

            # Ensure batch time dimension matches model.num_frames
            Tmodel = getattr(model, "num_frames", None)
            if Tmodel is not None and images.size(1) != Tmodel:
                if images.size(1) > Tmodel:
                    images = images[:, :Tmodel]
                else:
                    pad_frames = Tmodel - images.size(1)
                    pad = torch.zeros(images.size(0), pad_frames, images.size(2), images.size(3), images.size(4),
                                      device=images.device, dtype=images.dtype)
                    images = torch.cat([images, pad], dim=1)

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

            if verbose:
                for i in range(labels.size(0)):
                    print(f"Sample {sample_idx+i}: label={labels[i].item()} pred={pred[i].item()} conf={conf[i].item():.4f}")
                sample_idx += labels.size(0)

    acc = correct / max(1, total)
    mean_conf = total_conf / max(1, total)
    per_class_acc = {c: (per_class_correct[c] / per_class[c]) if per_class[c] > 0 else 0.0
                     for c in sorted(per_class.keys())}
    return acc, mean_conf, per_class_acc, total


def build_test_loader(csv_path: str, max_seq_length: int, batch_size=8, num_workers=2):
    df = load_dataframe(csv_path)  # sorted & parsed dates
    dataset = TimeSeriesImageDataset(path=None, df=df, max_seq_length=max_seq_length, transform=None)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Discover checkpoints
    ckpt_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pth")))
    if not ckpt_paths:
        print(f"[error] No .pth files found in: {os.path.abspath(checkpoint_dir)}")
        return
    print(f"Found {len(ckpt_paths)} checkpoint(s) in {checkpoint_dir}.")

    # CSV stats
    df = load_dataframe(csv_path)
    csv_max_T = int(df.groupby("Field_ID").size().max()) if len(df) else DEFAULT_FALLBACK_T
    print(f"CSV: {csv_path} | fields={df['Field_ID'].nunique()} | max_seq_len={csv_max_T}")

    results = []  # (filename, T, samples, acc, mean_conf, per_class_acc)

    for ckpt_path in ckpt_paths:
        fname = os.path.basename(ckpt_path)
        print("\n" + "=" * 80)
        print(f"Evaluating checkpoint: {fname}")

        # Load checkpoint to CPU
        obj = torch.load(ckpt_path, map_location="cpu")
        state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj

        # Skip non time-series (e.g., ViT single-frame)
        if not is_timeseries_checkpoint(state):
            print(f"[skip] {fname} does not look like a time-series transformer (no time_embeddings).")
            continue

        # Decide T: prefer T from checkpoint; fallback to CSV max; otherwise DEFAULT_FALLBACK_T
        inferred_T = infer_num_frames_from_state(state, fallback=csv_max_T)
        if inferred_T is None:
            inferred_T = DEFAULT_FALLBACK_T
        print(f"Inferred num_frames T={inferred_T} (CSV max={csv_max_T})")

        # Build model for this T
        model = TimeSeriesTransformer(img_size=224, num_frames=inferred_T, num_classes=2).to(device)

        # Interpolate time embeddings if needed
        state = maybe_interpolate_time_embeddings(model, state)

        # Load weights (tolerant)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[warn] Missing keys: {missing}")
        if unexpected:
            print(f"[warn] Unexpected keys: {unexpected}")

        # Build loader (data padded to inferred_T; we also pad/truncate at batch-time if needed)
        test_loader = build_test_loader(csv_path, max_seq_length=inferred_T, batch_size=8, num_workers=2)

        # Evaluate
        acc, mean_conf, per_class_acc, n = evaluate(model, test_loader, device, verbose=False)
        print(f"Samples: {n} | Accuracy: {acc:.4f} | Mean confidence: {mean_conf:.4f}")
        for c, a in per_class_acc.items():
            print(f"  Class {c} accuracy: {a:.4f}")

        results.append((fname, inferred_T, n, acc, mean_conf, per_class_acc))

    # Summary
    print("\n" + "#" * 80)
    print("SUMMARY (time-series checkpoints only):")
    if not results:
        print("No time-series checkpoints evaluated.")
        return

    # Sort by accuracy desc, then mean_conf desc
    results_sorted = sorted(results, key=lambda x: (x[3], x[4]), reverse=True)
    for fname, T, n, acc, mean_conf, _ in results_sorted:
        print(f"{fname:40s} | T={T:2d} | N={n:3d} | Acc={acc:.4f} | MeanConf={mean_conf:.4f}")


if __name__ == "__main__":
    main()
