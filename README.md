
# Synthetic Household Travel Data (NHTS 2017) — CTGAN Starter

This repository scaffolds your **EEE6778 Deliverable 1** project: *Synthetic Household Travel Data Generation using Conditional Tabular GAN (CTGAN)*.
This project explores how Conditional Tabular GAN (CTGAN) can generate synthetic but realistic household travel microdata using the 2017 National Household Travel Survey (NHTS).

By learning statistical relationships among key attributes – such as household size, vehicle ownership, income level, and daily trip count – the model can produce privacy-preserving datasets that mirror real mobility patterns while enabling data sharing for transportation research and policy analysis.

## Repo Structure
```
data/
  raw/                # put original NHTS CSVs here (not committed)
  sample/             # small samples for quick tests
notebooks/
  setup.ipynb         # environment, data loading, first EDA plot
src/
  data_prep.py        # cleaning & preprocessing helpers (rules 2,4,5)
  train_ctgan.py      # minimal CTGAN training script
  evaluate.py         # real vs synthetic evaluation (KS/MMD/corr)
ui/
  streamlit_app.py    # interactive demo to generate & compare data
results/
  # early figures or logs
docs/
  architecture.md     # data → model → inference → UI diagram
requirements.txt
environment.yml
```

## Quickstart
### Main Notebook
`notebooks/nhts_ctgan_training.ipynb` – loads NHTS 2017 data, applies cleaning rules (2 & 5), trains CTGAN, and evaluates real vs synthetic results.

### 1) Create environment & install
```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate nhts-ctgan

# Or pip
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Verify with the notebook
```bash
jupyter lab  # or: jupyter notebook
# Open notebooks/setup.ipynb and run all cells
```

### 3) Train CTGAN (toy run)
```bash
python src/train_ctgan.py --data data/sample/household_sample.csv --out results/ctgan_sample.pkl
```

### 4) Evaluate
```bash
python src/evaluate.py --real data/sample/household_sample.csv --synthetic results/synth_sample.csv --report results/report.json
```

### 5) Run the Streamlit UI
```bash
streamlit run ui/streamlit_app.py
```

## Data notes
- Place original NHTS CSVs under `data/raw/` (ignored by git).
- `data/sample/household_sample.csv` is a tiny synthetic-like subset for environment checks only.

## Dataset Information

Source: U.S. Department of Transportation – National Household Travel Survey 2017

Type: Tabular (mixed categorical and numerical)

Files used: Household and Person tables

Sample size (after cleaning): ≈ 84 000 households

Selected Features: HHVEHCNT, HHSIZE, HHFAMINC, URBRUR, CNTTDHH, R_AGE_IMP, DRIVER, WORKER

Cleaning Rules:
(2) Keep households where HHSIZE = person count
(5) Keep households whose members are all in-town (OUTOFTWN == 2, OUTCNTRY == 2)


## GitHub — First Commit
```bash
git init
git add .
git commit -m "Initial commit: EEE6778 CTGAN starter"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```
