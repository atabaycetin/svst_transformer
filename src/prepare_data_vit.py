import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# Load the CSV file
csv_path = "farm_data.csv"
df = pd.read_csv(csv_path)

# Define image transformations
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


class ViTDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform if transform else image_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Get image paths
        cig_path = row["CIG_Path"]
        evi_path = row["EVI_Path"]
        ndvi_path = row["NDVI_Path"]
        label = row["Infestation"]

        # Load images
        cig_img = self.load_image(cig_path)
        evi_img = self.load_image(evi_path)
        ndvi_img = self.load_image(ndvi_path)

        # Stack images into one tensor (Channels x Height x Width)
        image_tensor = torch.cat([cig_img, evi_img, ndvi_img], dim=0)

        return image_tensor, torch.tensor(label, dtype=torch.float32)

    def load_image(self, path):
        if path == "":
            return torch.zeros(3, 224, 224)  # Placeholder image
        return self.transform(Image.open(path).convert("RGB"))


# Create data
vit_dataset = ViTDataset(df)

# Dataloader
vit_dataloader = DataLoader(vit_dataset, batch_size=8, shuffle=True)

print("✅ ViT data is ready!")
