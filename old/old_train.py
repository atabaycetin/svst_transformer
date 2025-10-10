import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import TimesformerConfig, TimesformerModel
from prepare_data_timeseries_transformer import time_series_dataloader, MAX_SEQ_LENGTH

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define Transformer Model
class TimeSeriesTransformer(nn.Module):
    def __init__(self, img_size=224, num_frames=MAX_SEQ_LENGTH, num_classes=2):
        super(TimeSeriesTransformer, self).__init__()
        self.config = TimesformerConfig(
            image_size=img_size,
            num_frames=num_frames,
            patch_size=16,
            num_channels=9,  # 3 indices * 3 channels each
            num_classes=num_classes
        )
        self.model = TimesformerModel(self.config)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, x):
        outputs = self.model(x)
        # Pool over the time dimension (dim=1)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        x = self.classifier(pooled_output)
        return x

# Initialize Model
model = TimeSeriesTransformer().to(device)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in time_series_dataloader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels.long())  # Ensure labels are long-type for cross-entropy
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(time_series_dataloader)
    epoch_acc = correct / total

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

print("✅ Model training complete!")

# Save Model
torch.save(model.state_dict(), "time_series_transformer.pth")
print("✅ Model saved as time_series_transformer.pth")
