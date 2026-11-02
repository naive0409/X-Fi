"""csi_jepa: unimodal task-agnostic CSI foundation base (Phase A1, PROJECT_LOG §9).

WiFi-JEPA-style structured (link, time) tokenization of raw XRF55 WiFi CSI (270, 1000)
-> 9×10 = 90-token grid -> pre-LN ViT encoder trained from scratch with the same JEPA
objective proven in rounds 1-5 (EMA target encoder + predictor + SmoothL1 latent loss).
Replaces the HAR-supervised frozen ResNet extractor front-end — the residual source of
task specificity in the rounds-1-5 stack.
"""
