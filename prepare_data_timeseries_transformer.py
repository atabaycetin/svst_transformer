import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

MAX_SEQ_LENGTH = None

def load_dataframe(path):
    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(by=["Field_ID", "Date"])

    return df

def get_transformations():
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    return image_transform

class TimeSeriesImageDataset(Dataset):
    def __init__(self, path, max_seq_length, df=None, transform=None):
        if df is None and path is not None:
            df = pd.read_csv(path)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values(by=["Field_ID", "Date"])
        self.df = df
        self.max_seq_length = max_seq_length
        self.transform = transform if transform else get_transformations()
        self.field_ids = df["Field_ID"].unique()

    def __len__(self):
        return len(self.field_ids)

    def __getitem__(self, idx):
        field_id = self.field_ids[idx]
        field_data = self.df[self.df["Field_ID"] == field_id].sort_values("Date")

        sequences = []
        labels = []

        for i in tqdm(range(len(field_data)), desc=f"Loading Field {field_id}", leave=False):
            cig_path = field_data.iloc[i]["CIG_Path"]
            evi_path = field_data.iloc[i]["EVI_Path"]
            ndvi_path = field_data.iloc[i]["NDVI_Path"]
            label = field_data.iloc[i]["Infestation"]

            cig_img = self.load_image(cig_path)   # shape: [3, 224, 224]
            evi_img = self.load_image(evi_path)   # shape: [3, 224, 224]
            ndvi_img = self.load_image(ndvi_path)   # shape: [3, 224, 224]

            img_tensor = torch.cat([cig_img, evi_img, ndvi_img], dim=0)
            sequences.append(img_tensor)
            labels.append(label)

        sequences = torch.stack(sequences) if len(sequences) > 0 else torch.zeros(1, 9, 224, 224)

        if sequences.shape[0] < self.max_seq_length:
            padding = torch.zeros(self.max_seq_length - sequences.shape[0], 9, 224, 224)
            sequences = torch.cat([sequences, padding], dim=0)

        return sequences, torch.tensor(labels[-1], dtype=torch.float32)

    def load_image(self, path):
        if not path or not os.path.exists(path):
            return torch.zeros(3, 224, 224)
        return self.transform(Image.open(path).convert("RGB"))

def main():
    global MAX_SEQ_LENGTH

    path = "farm_data.csv"

    df = load_dataframe(path)

    MAX_SEQ_LENGTH = df.groupby("Field_ID").size().max()

    image_transform = get_transformations()

    time_series_dataset = TimeSeriesImageDataset(path=path, df=df, max_seq_length=MAX_SEQ_LENGTH, transform=image_transform)

    time_series_dataloader = DataLoader(time_series_dataset, batch_size=8, shuffle=True)

    print("✅ Time-Series Transformer dataset is ready and padded!")

if __name__ == "__main__":
    main()
