import torch
from prepare_data_timeseries_transformer import TimeSeriesImageDataset
from train_time_series_transformer import TimeSeriesTransformer, MAX_SEQ_LENGTH
import torch.nn.functional as F
from torch.utils.data import DataLoader

"""
switch train_time_series_transformer if statement to 0 to test it
"""

def main():
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate your model with the same configuration as during training
    model = TimeSeriesTransformer(img_size=224, num_frames=MAX_SEQ_LENGTH, num_classes=2)
    model.to(device)

    # Load the saved state dictionary
    model.load_state_dict(torch.load("time_series_transformer_v1.pth", map_location=device))

    # Set the model to evaluation mode
    model.eval()

    # Prepare your test dataset and DataLoader (make sure you use the same dataset splits as before)
    # Example: (assuming TimeSeriesImageDataset can load the test data)
    test_dataset = TimeSeriesImageDataset(df=None)  # Or however you load your test set
    # You might have already split it; adjust this as needed.
    # For demonstration, here's how you might create a DataLoader:

    BATCH_SIZE = 8
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Compute test accuracy
    correct_test = 0
    total_test = 0
    total_confidence = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)

            # Convert logits to probabilities
            probabilities = F.softmax(outputs, dim=1)
            # Get the predicted class and its confidence
            confidence, predicted = torch.max(probabilities, dim=1)

            # Now 'predicted' contains the predicted labels,
            # and 'confidence' contains the probability (confidence) for that prediction.
            for i in range(len(predicted)):
                print(f"Sample {i}: Predicted label: {predicted[i].item()}, Confidence: {confidence[i].item():.4f}")
                total_confidence += confidence[i].item()

    test_acc = correct_test / total_test
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Total Test Confidence: {total_confidence / total_test:.4f}")


if __name__ == "__main__":
    main()
