---
title: Fashion MNIST Dashboard
emoji: 👕
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Dashboard for the Fashion MNIST classifier API
---

# Fashion MNIST Dashboard

Static React dashboard. It has no model of its own — every prediction, metric
and retraining run comes from the API Space, whose origin is baked in at build
time via `VITE_API_BASE`.

Because this is a separate deployment from the API, the two sleep on
independent idle timers: opening this page can find the API still asleep. The
dashboard handles that with a wake-up screen and keeps requesting until the API
answers, and the repository's `keep-spaces-awake` workflow stops it happening
in the first place. Deploying the single-container image to both Spaces avoids
the situation entirely.

Source: https://github.com/Chol1000/fashion-mnist-mlops
