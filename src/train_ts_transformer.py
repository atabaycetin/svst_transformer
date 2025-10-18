import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from transformers import TimesformerConfig, TimesformerModel
from torch.utils.data import DataLoader, random_split, Dataset

from prepare_data_ts_transformer import (
    TimeSeriesImageDataset,
    get_transformations,
    load_dataframe,
)

# Set random seed for reproducibility
torch.manual_seed(42)

# Set path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = "farm_data.csv"

# Set default fallback
DEFAULT_NUM_FRAMES = 8

class AugmentedTimeSeriesDataset(Dataset):
    """
    For each sample (sequence [T, 9, 224, 224]), return 3 variants:
      0: original, 1: vertical flip, 2: horizontal flip.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset) * 3

    def __getitem__(self, idx):
        base_idx = idx // 3
        aug_idx = idx % 3
        sequence, label = self.base_dataset[base_idx]  # [T, 9, 224, 224]

        if aug_idx == 0:
            return sequence, label

        T = sequence.shape[0]
        out = torch.zeros_like(sequence)
        for t in range(T):
            frame = sequence[t]  # [9, 224, 224]
            # split into 3×RGB blocks to avoid mixing the indices
            p1, p2, p3 = frame[0:3], frame[3:6], frame[6:9]
            if aug_idx == 1:      # vertical
                p1, p2, p3 = torch.flip(p1, [1]), torch.flip(p2, [1]), torch.flip(p3, [1])
            else:                 # horizontal
                p1, p2, p3 = torch.flip(p1, [2]), torch.flip(p2, [2]), torch.flip(p3, [2])
            out[t] = torch.cat([p1, p2, p3], dim=0)
        return out, label


class TimeSeriesTransformer(nn.Module):
    def __init__(self, img_size=224, num_frames=None, num_classes=2):
        super().__init__()
        if num_frames is None:
            num_frames = DEFAULT_NUM_FRAMES
        self.num_frames = int(num_frames)

        self.config = TimesformerConfig(
            image_size=img_size,
            num_frames=self.num_frames,
            patch_size=16,
            num_channels=9,  # CIG/EVI/NDVI (3×RGB)
            # num_labels isn't used by TimesformerModel; we add our own classifier
        )
        self.model = TimesformerModel(self.config)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, x):  # x: [B, T, 9, 224, 224]
        outputs = self.model(x)  # BaseModelOutput, last_hidden_state shape: [B, T, P+1, H] or [B, L, H] depending on HF version
        # Robust pooling: flatten all tokens except batch, then mean
        last = outputs.last_hidden_state
        # If last is [B, T, tokens, H] → mean over (T,tokens); if [B, tokens, H] → mean over tokens
        if last.dim() == 4:
            pooled = last.mean(dim=(1, 2))
        else:
            pooled = last.mean(dim=1)
        return self.classifier(pooled)


def main():

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataframe & max sequence length
    df = load_dataframe(csv_path)
    MAX_T = int(df.groupby("Field_ID").size().max()) if len(df) else DEFAULT_NUM_FRAMES

    # Dataset & splits
    base_dataset = TimeSeriesImageDataset(
        df=df,
        path=None,                     # df already provided
        max_seq_length=MAX_T,          # ensure proper padding length
        transform=get_transformations()
    )

    dataset_size = len(base_dataset)
    train_size = int(0.6 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    train_ds, val_ds, test_ds = random_split(base_dataset, [train_size, val_size, test_size])
    print(f"Dataset split: {train_size} train, {val_size} val, {test_size} test samples.")

    # Augment training set
    train_ds = AugmentedTimeSeriesDataset(train_ds)

    # Loaders
    BATCH_SIZE = 8
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Model / loss / optim
    model = TimeSeriesTransformer(num_frames=MAX_T).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-8)

    # Train
    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            images = images.to(device)
            labels = labels.long().to(device)

            # If any sample length differs, truncate/pad to MAX_T (safety)
            if images.size(1) != model.num_frames:
                Tmodel = model.num_frames
                if images.size(1) > Tmodel:
                    images = images[:, :Tmodel]
                else:
                    pad_frames = Tmodel - images.size(1)
                    pad = torch.zeros(images.size(0), pad_frames, images.size(2), images.size(3), images.size(4),
                                      device=images.device, dtype=images.dtype)
                    images = torch.cat([images, pad], dim=1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(1, len(train_loader))
        train_acc = correct / max(1, total)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.long().to(device)

                if images.size(1) != model.num_frames:
                    Tmodel = model.num_frames
                    if images.size(1) > Tmodel:
                        images = images[:, :Tmodel]
                    else:
                        pad_frames = Tmodel - images.size(1)
                        pad = torch.zeros(images.size(0), pad_frames, images.size(2), images.size(3), images.size(4),
                                          device=images.device, dtype=images.dtype)
                        images = torch.cat([images, pad], dim=1)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss /= max(1, len(val_loader))
        val_acc = correct_val / max(1, total_val)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    print("✅ Training complete!")

    # Test
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.long().to(device)

            if images.size(1) != model.num_frames:
                Tmodel = model.num_frames
                if images.size(1) > Tmodel:
                    images = images[:, :Tmodel]
                else:
                    pad_frames = Tmodel - images.size(1)
                    pad = torch.zeros(images.size(0), pad_frames, images.size(2), images.size(3), images.size(4),
                                      device=images.device, dtype=images.dtype)
                    images = torch.cat([images, pad], dim=1)

            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct_test += (preds == labels).sum().item()
            total_test += labels.size(0)

    test_acc = correct_test / max(1, total_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save
    checkpoint_dir = os.path.join(ROOT, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "ts_transformer.pth")

    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")


if __name__ == "__main__":
    main()
