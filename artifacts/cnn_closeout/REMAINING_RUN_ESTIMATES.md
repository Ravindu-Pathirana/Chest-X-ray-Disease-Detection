# Remaining Checkpoint-Dependent Run Estimates

These are planning ranges, not guarantees. They assume one Kaggle-class GPU
(T4/P100 or similar), 224x224 inputs, cached/preprocessed data, and working
checkpoints. Data download, queue time, notebook recovery, and debugging are
not included unless stated.

| Remaining work | Minimum scope | Rough elapsed GPU time | All three seeds | Repository readiness |
|---|---|---:|---:|---|
| DenseNet + ResNet background counterfactuals | A0/selected, seed 42, zero/shuffle/noise | 1–3 h total | 3–8 h | Evaluation code exists; checkpoints required. |
| DenseNet + ResNet temperature scaling | Validation + test for A0/selected | 0.5–1.5 h total | 1.5–4 h | Harness exists; checkpoints or saved validation logits required. |
| Lung-versus-background occlusion | Six CNNs, seed 42 | 1–3 h | 3–8 h | Small extension to counterfactual code; checkpoints required. |
| RSNA external inference | Six CNNs, seed 42, about 26.7k studies | 3–8 h | 8–24 h | Dataset prepared; label mapping and final runner still required. |
| Additional Grad-CAM-style method | Six CNNs, fixed 1,000-image subset | 2–6 h | 6–18 h | New method integration and checkpoints required. |

## Important schedule risk

No `.pt`, `.pth`, or `.ckpt` model files are present in the repository. If the
trained weights cannot be recovered, the inference estimates above do not
apply. Retraining the selected baseline/attention pairs across three seeds can
reasonably require roughly 24–72 aggregate GPU-hours, depending on GPU type,
early stopping, caching, and notebook stability.

## Recommended stopping rule

The existing-data closeout is sufficient for the CNN anatomical-localisation
study. Run checkpoint-dependent work only if the weights are recovered quickly.
Prioritize, in order: (1) DenseNet/ResNet counterfactuals, (2) calibrated ECE,
(3) one occlusion experiment, and (4) RSNA. Do not allow ViT adaptation or a
new saliency family to delay the CNN result freeze.
