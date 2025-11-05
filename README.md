
# Ancient-to-Film GAN
This project converts **ancient Chinese landscape paintings** into **modern film-style photographs** using a **GAN-based image-to-image translation pipeline** (CycleGAN / Pix2Pix).  
Deliverable 1 demonstrates dataset loading, environment setup, a tiny generator test, and an interactive demo UI.
Model Design

Unpaired translation (default): CycleGAN — translates between Domain A (ancient drawings) and Domain B (film-style photos) without one-to-one pairing.
Paired translation (optional): Pix2Pix — supervised learning for aligned image pairs.
The current prototype uses a Tiny Generator as a proof of concept. Future iterations will integrate the full CycleGAN architecture with adversarial, cycle-consistency, and identity losses.

Dataset

Domain A: Ancient Chinese landscape paintings
Domain B: Modern film-style photographs
All images are resized to 256×256 and normalized to [-1, 1].
A small subset (~50–100 images per domain) is included for demonstration; full datasets are excluded from GitHub for size and copyright reasons.

Results

After running the toy demo, generated samples will appear in results/samples/, such as toy_step_000.png and toy_step_001.png. These verify that the data pipeline, model, and output generation work correctly.

Responsible AI Reflection

Authenticity & Cultural Context: The generated outputs are artistic reinterpretations of ancient art, not authentic historical photographs.
Fair Use: All art sources are credited; outputs are for academic demonstration only and not for commercial use.
Environmental Impact: Training and inference are kept lightweight (≤ 50 iterations) to reduce computational load.


EEE6778_CTGAN_NHTS/
├─ data/
│  ├─ A/  ← Ancient drawings
│  └─ B/  ← Film-style photos
├─ notebooks/
│  └─ setup.ipynb          # Environment + data check + TinyGen forward test
├─ src/
│  ├─ datasets.py
│  ├─ models/
│  │   └─ cyclegan.py
│  ├─ trainers/
│  │   └─ train_cyclegan.py
│  └─ __init__.py
├─ ui/
│  └─ streamlit_app.py     # Gradio UI demo
├─ docs/
│  ├─ architecture.md      # Flowchart + component overview
│  └─ wireframe.png        # Optional UI sketch
├─ results/
│  └─ samples/             # Toy outputs / training previews
├─ requirements.txt
└─ README.md
