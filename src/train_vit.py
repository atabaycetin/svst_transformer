import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from transformers import ViTForImageClassification, ViTConfig
from prepare_data_vit import vit_dataloader, ViTDataset, df


# Set random seed for reproducibility
torch.manual_seed(42)

# Load data from DataFrame
full_dataset = ViTDataset(df)

# Split data: 70% train, 20% validation, 10% test
dataset_size = len(full_dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.2 * dataset_size)
test_size = dataset_size - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
print(f"Dataset split: {train_size} train, {val_size} val, {test_size} test samples.")

# DataLoaders
BATCH_SIZE = 8
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define ViT Model
class ViTModel(nn.Module):
    def __init__(self, img_size=224, num_classes=2):
        super(ViTModel, self).__init__()
        self.config = ViTConfig(
            image_size=img_size,
            num_channels=9,  # 3 indices concatenated (CIG, EVI, NDVI)
            num_labels=num_classes,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12
        )
        self.model = ViTForImageClassification(self.config)

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs.logits  # Use logits for classification

# Initialize model
model = ViTModel().to(device)

# Define Loss, Optimizer, and Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

# Training Loop with validation
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
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
            images, labels = images.to(device), labels.to(device).long()
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = correct_val / total_val

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    scheduler.step()  # Update learning rate

print("✅ Training complete!")

# Evaluate on test set
model.eval()
correct_test = 0
total_test = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device).long()
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct_test += (predicted == labels).sum().item()
        total_test += labels.size(0)

test_acc = correct_test / total_test
print(f"Test Accuracy: {test_acc:.4f}")

# Save the model
torch.save(model.state_dict(), "prot5_vit_model.pth")
print("✅ Model saved as prot5_vit_model.pth")
