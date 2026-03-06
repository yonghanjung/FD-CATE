---
title: FD-CATE Demo
emoji: 📊
colorFrom: red
colorTo: yellow
sdk: gradio
python_version: "3.10"
sdk_version: "5.50.0"
app_file: app.py
pinned: true
tags:
- causal-inference
- front-door
- heterogeneous-treatment-effects
- unmeasured-confounding
- python
---

# FD-CATE Demo

Personalized causal inference under unmeasured confounding via front-door identification.

This Space demonstrates the core idea of [Debiased Front-Door Learners for Heterogeneous Effects](https://arxiv.org/abs/2509.22531): when treatment and outcome share hidden confounders but an observed mediator makes front-door identification plausible, we can still estimate heterogeneous treatment effects.

## What this demo does

- Generates a synthetic front-door dataset
- Fits one of `FD-PI`, `FD-DR`, or `FD-R`
- Visualizes the estimated treatment-effect distribution
- Optionally runs a compact benchmark summary
- Returns a downloadable artifact bundle

## Recommended defaults

- `n = 300`
- `d = 6`
- `method = FD-DR`
- `nuisance learner = xgb`

## Links

- Paper: <https://arxiv.org/abs/2509.22531>
- Package: <https://pypi.org/project/fd-cate/>
- Code: <https://github.com/yonghanjung/FD-CATE>
