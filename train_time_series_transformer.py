import torch
import torch.nn as nn
import torch.optim as optim
from transformers import TimesformerConfig, TimesformerModel
from torch.utils.data import DataLoader, random_split, Dataset
from prepare_data_timeseries_transformer import TimeSeriesImageDataset, MAX_SEQ_LENGTH


# Set random seed for reproducibility
torch.manual_seed(42)

# Load the full dataset from CSV using the dataset class
full_dataset = TimeSeriesImageDataset(df=None)

# Alternatively, if you exported time_series_dataset in your prepare file, you can:
# from prepare_data_timeseries_transformer import time_series_dataset
# full_dataset = time_series_dataset

# Split dataset: 60% train, 15% validation, 25% test
dataset_size = len(full_dataset)
train_size = int(0.6 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
print(f"Dataset split: {train_size} train, {val_size} val, {test_size} test samples.")


# Define an augmentation wrapper for the training set
class AugmentedTimeSeriesDataset(Dataset):
    """
    For each sample (a sequence of frames with shape [T, 9, 224, 224]),
    return three versions:
      0: Original,
      1: Vertically flipped (flip height of each 3-channel segment),
      2: Horizontally flipped (flip width).
    """

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset) * 3

    def __getitem__(self, idx):
        base_idx = idx // 3
        aug_idx = idx % 3
        sequence, label = self.base_dataset[base_idx]  # sequence shape: [T, 9, 224, 224]

        if aug_idx == 0:
            # Original
            return sequence, label
        elif aug_idx == 1:
            # Vertical flip: For each frame, flip along height (dim=1) separately for each 3-channel block.
            seq_v = torch.zeros_like(sequence)
            T = sequence.shape[0]
            for t in range(T):
                frame = sequence[t]  # shape: [9, 224, 224]
                part1 = frame[0:3, :, :]
                part2 = frame[3:6, :, :]
                part3 = frame[6:9, :, :]
                part1_v = torch.flip(part1, dims=[1])
                part2_v = torch.flip(part2, dims=[1])
                part3_v = torch.flip(part3, dims=[1])
                seq_v[t] = torch.cat([part1_v, part2_v, part3_v], dim=0)
            return seq_v, label
        elif aug_idx == 2:
            # Horizontal flip: For each frame, flip along width (dim=2)
            seq_h = torch.zeros_like(sequence)
            T = sequence.shape[0]
            for t in range(T):
                frame = sequence[t]
                part1 = frame[0:3, :, :]
                part2 = frame[3:6, :, :]
                part3 = frame[6:9, :, :]
                part1_h = torch.flip(part1, dims=[2])
                part2_h = torch.flip(part2, dims=[2])
                part3_h = torch.flip(part3, dims=[2])
                seq_h[t] = torch.cat([part1_h, part2_h, part3_h], dim=0)
            return seq_h, label


# Wrap the training dataset with augmentation
augmented_train_dataset = AugmentedTimeSeriesDataset(train_dataset)

# Create DataLoaders
BATCH_SIZE = 8
train_loader = DataLoader(augmented_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define the Time-Series Transformer model
class TimeSeriesTransformer(nn.Module):
    def __init__(self, img_size=224, num_frames=MAX_SEQ_LENGTH, num_classes=2):
        super(TimeSeriesTransformer, self).__init__()
        self.config = TimesformerConfig(
            image_size=img_size,
            num_frames=num_frames,
            patch_size=16,
            num_channels=9,  # 3 indices concatenated, each 3 channels
            num_classes=num_classes
        )
        self.model = TimesformerModel(self.config)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, x):
        outputs = self.model(x)  # outputs: BaseModelOutput, with last_hidden_state
        # Pool over the time dimension (dim=1)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        x = self.classifier(pooled_output)
        return x


# Initialize model
model = TimeSeriesTransformer().to(device)

# Define Loss, Optimizer, and (optionally) a scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-8)
# Optionally, use a scheduler:
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)


# switch train_time_series_transformer if statement to 0 to test it
if 0:
    # Training Loop with validation
    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.long())
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
        val_loss /= len(val_loader)
        val_acc = correct_val / total_val

        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Optionally step the scheduler:
        # scheduler.step()

    print("✅ Training complete!")

    # Evaluate on test set
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)
    test_acc = correct_test / total_test
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save the model
    torch.save(model.state_dict(), "prot7_time_series_transformer.pth")
    print("✅ Model saved as prot7_time_series_transformer.pth")
