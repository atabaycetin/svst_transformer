# SVST – Sentinel-2 Vegetation Time-Series (ViT / TimeSformer)

Educational project exploring whether **time-series of vegetation indices** can help flag **possible pest infestation** in farm fields.

> **Important:** This repository is for learning and demonstration.  
> I used Sentinel-2 imagery to derive **EVI**, **NDVI**, and **CIG** (three-channel RGB renders per index). I could **not** confirm with 100% certainty which fields were actually infested during the supposed infestation period. Treat any results as **heuristic** rather than ground truth.


## Background

This project was developed for the **SVST Hackathon** ([svst.it/hackathon](https://www.svst.it/hackathon/)). The hackathon’s goal was primarily to ideate a **business concept**, not to deliver a production-ready product; we decided to also prototype a **working system** alongside the business idea. It was a fun and very **educational** experience.

---

## What’s inside

- **Data schema**: a single CSV (`farm_data.csv`) listing frames per field over time with paths to three images:
  - `CIG_Path`, `EVI_Path`, `NDVI_Path` (each a 3-channel 224×224 PNG/JPG)
  - labels are **binary** (`Infestation` ∈ {0,1}) and represent the **field state at/near the last date** of its sequence
- **Models**
  - **Time-Series Transformer** using HuggingFace **Timesformer** backbone (video transformer) over sequences of stacked 9-channel frames (3 indices × 3 channels)
  - **ViT baseline** on single frames (9-channel input created by concatenating CIG/EVI/NDVI)
- **Training & evaluation scripts**
  - `train_time_series_transformer.py` – trains a time-series model (import-safe)
  - `train_vit.py` – trains the 9-channel ViT baseline  
  - `test_time_series_transformer.py` – auto-discovers `*.pth` in a folder and evaluates each (also works without my private checkpoints)

> **No real data or checkpoints are included** in this repository.  
> The code is structured so you can plug in your own data following the CSV format below.

---

## Repository structure

```
.
├── src/
│   ├── data/
│   │   ├── CIG_infected/            # example raw tiles (images)
│   │   ├── CIG_non-infected/
│   │   ├── EVI_infected/
│   │   ├── EVI_non-infected/
│   │   ├── NDVI_infected/
│   │   └── NDVI_non-infected/
│   ├── farm_data.csv                # data manifest used by loaders
│   ├── prepare_data_ts_transformer.py
│   ├── prepare_data_vit.py
│   ├── setup_data.py
│   ├── test_ts_transformer.py
│   ├── test_vit.py
│   ├── train_ts_transformer.py
│   └── train_vit.py
├── checkpoints/                     # (gitignored) saved *.pth; created at runtime
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

You may also add a small synthetic data generator to let others run the pipeline without real data.

---

## Environment

```
python >= 3.10
torch >= 2.1
torchvision >= 0.16
transformers >= 4.40
tqdm, pandas, Pillow, pyyaml
```

Install:
```bash
pip install -r requirements.txt
```

---

## Data format (`farm_data.csv`)

Each row is one **frame** of a field at a given date. The evaluator/trainer groups by `Field_ID` and sorts by `Date`.

| Date       | Field_ID | CIG_Path            | EVI_Path            | NDVI_Path           | Infestation |
|------------|----------|---------------------|---------------------|---------------------|-------------|
| 2024-01-05 | 12       | data/CIG/F12_T0.png | data/EVI/F12_T0.png | data/NDVI/F12_T0.png| 0 or 1      |

**Notes**
- Paths should be valid on your machine (absolute or relative).
- Images are loaded, resized to **224×224**, and concatenated channel-wise: **[3(CIG)+3(EVI)+3(NDVI) = 9 channels]**.
- For time-series, sequences are **padded/truncated** to a uniform length internally (the code infers the length from your data or checkpoint).

---

## Quickstart (evaluation without my checkpoints)

1) Prepare your own `farm_data.csv` with the schema above and put images in place.  
2) Drop any time-series checkpoints (`*.pth`) into `checkpoints/`.  
3) Run:
   ```bash
   python test_time_series_transformer.py
   ```
   The script:
   - auto-discovers `checkpoints/*.pth`
   - infers the sequence length from each checkpoint
   - evaluates all time-series checkpoints on your CSV
   - prints per-class accuracy and a summary table

> The script **skips** ViT checkpoints automatically (those don’t have time embeddings).  
> If you want a similar tester for ViT, adapt `prepare_data_vit.py` + your ViT model to a `test_vit.py`.

---

## Training

### Time-Series Transformer
```bash
python train_time_series_transformer.py
```
Key implementation details:
- Uses HuggingFace **Timesformer** with `num_channels=9`, `img_size=224`, `patch_size=16`
- A simple classifier head on top of pooled tokens
- Training/val/test splits are done **by field** (no frame leakage)
- Augmentations: simple horizontal/vertical flips (index-aware)
- Saves checkpoint as `prot7_time_series_transformer.pth` (gitignored)

### ViT Baseline
```bash
python train_vit.py
```
- ViT with **9-channel** input (stacked indices), 224×224
- Cross-entropy objective

---

## Important limitations & caveats

- **Label uncertainty.** I could **not** verify with full certainty whether individual fields were truly infested during the target period. Labels should be treated as **noisy/weak**.
- **Imbalance.** Real-world datasets often have many more non-infested fields than infested ones, which can cause **majority-class collapse**. Consider **class weights**, **focal loss**, or **balanced splits** if you extend this work.
- **Do not use for decisions.** This repository is not a scientific tool nor an operational detection system. It’s a **first ML project** and a **learning exercise**.

---

## How to adapt/extend

- Swap the label policy (currently uses the **last date’s** label per field) for:
  - *Any-positive* across the sequence, or
  - *Majority/rolling* rule over the last *k* frames
- Add **balanced metrics** (balanced accuracy, F1 per class) and a **confusion matrix**
- Try **class-weighted CE**, **focal loss**, or **label smoothing**
- Add an **ensemble tester** to rank fields by **disagreement/uncertainty** when labels are unreliable
- Pretrain encoders with **self-supervised** methods on your imagery, then fine-tune

---

## Why 9 channels?

Each frame concatenates three 3-channel index renders:
- **CIG** (RGB render; a color/vegetation proxy)  
- **EVI** (Enhanced Vegetation Index, RGB render)  
- **NDVI** (Normalized Difference Vegetation Index, RGB render)  

This creates a **9-channel tensor** per timestamp that the models can ingest directly.

---

## License

Choose the license that fits your needs (e.g., **MIT**).  
Add a `LICENSE` file at the root.

---

## Acknowledgements

- Sentinel-2 data (Copernicus / ESA)  
- HuggingFace Transformers (Timesformer)  
- PyTorch & TorchVision

---

### Final note

This was my **first machine learning project**. It was challenging and fun, and the goal here is to share the code structure and lessons—not to provide a production system or definitive claims about pest infestation. If you build on this, please validate against **trusted labels** and domain expertise.

