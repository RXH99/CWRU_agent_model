# CWRU Fault Diagnosis Agent

A full-stack intelligent fault diagnosis project for rolling bearings, combining **few-shot learning**, **model interpretability**, and **AI agent**.

> For graduate research: Few-shot fault diagnosis + Explainable AI  
> Technical stack: PyTorch + Prototypical Networks + Integrated Gradients + LangChain

---

## Project Pipeline

```
Raw Vibration Signal
    ↓
1D-CNN Feature Extractor (pretrained on full CWRU data)
    ↓
Prototypical Networks (5-shot classification)
    ↓
Integrated Gradients (input attribution heatmap)
    ↓
AI Agent (LangChain) → auto diagnosis + report generation
```

## Results

| Experiment | Accuracy |
|-----------|----------|
| 1D-CNN baseline (full data) | 100% |
| Prototypical Network (5-shot) | 100% |
| Cross-load 0hp → 1hp | 100% |
| Cross-load 0hp → 2hp | 100% |
| Cross-load 0hp → 3hp | 100% |

IG attribution maps show the model focuses on class-specific temporal impact patterns, confirming physically meaningful learning rather than global statistic memorization.

## Project Files

| Script | Purpose |
|--------|---------|
| `step2_preprocess.py` | Load CWRU .mat data, sliding window segmentation |
| `step3_train.py` | Train 1D-CNN baseline (full data) |
| `step4_prototypical.py` | Prototypical network few-shot training |
| `step5_integrated_gradients.py` | IG attribution for model interpretability |
| `step5b_analyze_ig.py` | Quantitative analysis of attribution maps |
| `step6a_cross_load_test.py` | Cross-load generalization validation |

## Dataset

CWRU (Case Western Reserve University) Rolling Bearing Data Center  
- **Sampling rate:** 12 kHz (Drive End)  
- **Fault diameter:** 0.021 inches  
- **Classes:** Normal, Inner Race, Ball, Outer Race  
- **Loads:** 0hp, 1hp, 2hp, 3hp  
- Download: https://engineering.case.edu/bearingdatacenter

## Quick Start

```bash
# 1. Preprocess data
python step2_preprocess.py

# 2. Train 1D-CNN
python step3_train.py

# 3. Train Prototypical Network
python step4_prototypical.py

# 4. Generate IG attribution maps
python step5_integrated_gradients.py

# 5. Cross-load validation
python step6a_cross_load_test.py
```

## Requirements

```
torch>=2.0
numpy
scipy
matplotlib
scikit-learn
```

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Few-shot method | Prototypical Networks | Metric learning, stable under 5-shot |
| Interpretability | Integrated Gradients | Bypasses GAP layer limitation (Grad-CAM fails with global average pooling) |
| Feature extractor | 1D-CNN + pretrained | Transfer from full-data training |

## License

MIT
