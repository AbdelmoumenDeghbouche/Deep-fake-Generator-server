# GAN Face Generator — REST API

> A production-deployed REST API serving a TensorFlow Lite GAN generator that synthesizes photorealistic face images from random latent vectors.

---

## Overview

This project wraps a pre-trained **Generative Adversarial Network (GAN)** generator in a lightweight FastAPI server, enabling on-demand generation of synthetic face images via a simple HTTP endpoint.

The model takes a random noise vector as input and outputs a realistic face image — with no source image required. This demonstrates both **generative AI deployment** and **adversarial model serving** in a production-ready architecture.

---

## How It Works

```
Client Request (POST /generate_images?num_images=N)
        │
        ▼
┌──────────────────────────────────────┐
│  Sample N latent vectors             │
│  z ~ Normal(0, 0.45) ∈ ℝ³²          │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  TFLite GAN Generator                │
│  Input:  [batch=1, latent_dim=32]    │
│  Output: [1, H, W, 3] ∈ [-1, 1]     │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Post-processing                     │
│  • Scale [-1,1] → [0,255] uint8      │
│  • Convert to PIL RGB image          │
│  • Encode as PNG                     │
└──────────────┬───────────────────────┘
               │
               ▼
        PNG image bytes returned
```

---

## API Reference

### `GET /`
Health check — returns server status.

### `POST /generate_images`

**Query parameter:** `num_images` (int, default=1)

**Returns:** PNG image (or concatenated PNGs for multiple images)

```bash
# Generate 4 synthetic faces
curl -X POST "https://your-deployment/generate_images?num_images=4" \
     --output faces.png
```

---

## Technical Details

| Component | Detail |
|---|---|
| **Model format** | TensorFlow Lite (`.tflite`) — optimized for edge deployment |
| **Latent space** | 32-dimensional standard normal, scaled by 0.45 |
| **Output** | RGB images, 3 channels |
| **Inference** | `tf.lite.Interpreter` — CPU inference, no GPU required |
| **Deployment** | Vercel serverless via `@vercel/python` |
| **Framework** | FastAPI + Uvicorn |

---

## Stack

- **Python 3.x**
- **TensorFlow Lite** (inference only — no training code)
- **FastAPI** + **Uvicorn**
- **Pillow** (image encoding)
- **NumPy** (latent vector sampling)
- **Vercel** (serverless deployment)

---

## Run Locally

```bash
pip install fastapi uvicorn tensorflow pillow numpy
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Research Context

Generative models such as GANs raise important questions in AI security:
- **Synthetic media detection** — how can IDS/anomaly detection systems identify AI-generated content in network traffic?
- **Adversarial robustness** — GAN outputs can be used to craft adversarial examples that fool classifiers
- **Data augmentation** — synthetic data generation addresses the scarcity of labeled attack samples in IDS research (a core limitation of the CICIoMT2024 dataset used in the author's published work)

This project reflects the author's interest in the intersection of generative AI and security systems.
