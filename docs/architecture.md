
# Architecture Design — Ancient-to-Film GAN

## 1. Overview
This project converts **ancient drawings (Domain A)** into **modern film-style photos (Domain B)** using **GAN-based image-to-image translation**.

## 2. Data Flow
```mermaid
flowchart LR
    A[Domain A: Ancient Drawings] --> P[Preprocess]
    B[Domain B: Film-Style Photos] --> P
    P --> M{Model}
    M -->|Unpaired| C[CycleGAN]
    M -->|Paired| X[Pix2Pix]
    C --> I[Inference]
    X --> I
    I --> UI[Gradio UI]
    UI --> R[Results & Logs]
