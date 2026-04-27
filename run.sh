#!/bin/bash
# TAAC 2026 - Tencent UNI-REC Challenge (KDD Cup 2026)
# Unified Sequence + Feature Interaction for CVR Prediction

set -e

echo "============================================"
echo " TAAC 2026 - Unified Rec Model"
echo " Metric: AUC of ROC"
echo "============================================"

# ── Single GPU Training ──
# uv run python train.py

# ── Inference ──
# uv run python inference.py

echo "Done! Check checkpoints/ for results."
