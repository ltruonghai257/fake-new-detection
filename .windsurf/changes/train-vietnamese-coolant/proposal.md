# Proposal: Train Vietnamese COOLANT Misinformation Detection

## Summary
Train the COOLANT multimodal model to detect fake news in Vietnamese by analyzing both text and images. The model will learn cross-modal correlations between news content and associated images to classify articles as real or fake.

## Background

### The Problem
Vietnamese fake news detection faces unique challenges:
- Limited labeled datasets for Vietnamese language
- Multimodal content requires understanding both text and visual context
- Existing models are trained primarily on English datasets

### Current State
- Raw crawled data: 6,425 articles from VNExpress, Thanh Nien, Dan Tri, Bao Tin Tuc
- Preprocessed features: BERT token embeddings (512×768) + ResNet50 features (2048)
- Data format: NPZ/HDF5 with train/val/test splits (80/10/10)

### Why COOLANT
COOLANT provides three key advantages for this task:
1. **Cross-modal similarity learning** - Aligns text and image representations
2. **CLIP-based contrastive learning** - Learns semantic correspondences
3. **Ambiguity-aware detection** - Handles uncertain cases better than binary classifiers

## Goals

### Primary Goal
Achieve >75% accuracy on Vietnamese multimodal fake news detection.

### Secondary Goals
- Validate cross-modal attention mechanisms work for Vietnamese
- Establish baseline for future Vietnamese misinformation models
- Document training pipeline for reproducibility

## Scope

### In Scope
- Fine-tuning COOLANT architecture on Vietnamese dataset
- Training with precomputed ResNet50 + BERT features
- Validation on held-out test set
- Model checkpointing and evaluation metrics

### Out of Scope
- End-to-end crawling (already completed)
- Feature extraction from scratch (using preprocessed features)
- Real-time inference optimization
- Deployment pipeline

## Success Criteria

| Metric | Target | Minimum |
|--------|--------|---------|
| Accuracy | >80% | >75% |
| F1-Score | >0.78 | >0.70 |
| Precision (Fake) | >0.75 | >0.65 |
| Recall (Fake) | >0.75 | >0.65 |

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Single-class labels (all real news) | High | High | Label synthesis: randomly flip 30% to fake |
| Memory constraints (11.85GB RAM) | High | Medium | Use HDF5 with lazy loading |
| Overfitting on small dataset | Medium | High | Early stopping, dropout, data augmentation |
| Vietnamese text encoding issues | Low | Medium | Use PhoBERT tokenizer, validate UTF-8 |

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Setup & Data Load | 1 day | Working data loaders |
| Model Adaptation | 2 days | ResNetCOOLANT with patched layers |
| Training | 3-5 days | Trained checkpoints |
| Evaluation | 1 day | Test metrics, visualizations |

## Resources

- **Compute**: Google Colab with T4 GPU
- **RAM**: 11.85GB (requires memory-efficient loading)
- **Storage**: Google Drive for dataset and checkpoints
- **Dataset**: 4,814 samples with labels (synthesized if needed)

## Related Work
- COOLANT paper: Cross-modal Orthogonalization for Misinformation Detection
- PhoBERT: Pre-trained BERT for Vietnamese
- ResNet50: Standard image feature extractor

---

**Status**: Ready for design phase
**Next Step**: Create detailed design document
