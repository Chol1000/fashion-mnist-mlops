---
title: Fashion MNIST MLOps
emoji: 👕
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: MobileNetV2 garment classifier with a live retraining pipeline
---

# Fashion MNIST MLOps

Dashboard and API in one container. Open the root URL for the dashboard, or
`/docs` for the interactive API reference.

- `GET  /health` — model readiness and database stats
- `POST /predict` — classify from 784 pixel values
- `POST /predict/image` — classify from an uploaded image
- `POST /retrain` — fine-tune on uploaded samples

Source: https://github.com/Chol1000/fashion-mnist-mlops
