# Kaggle execution handoff — no account connection yet

This repo's `explainable-AI` branch is the code source. The unrelated `kaggle`
Git remote is **not** used for commits or synchronization. Kaggle notebook
execution would require an authenticated Kaggle account, notebook access and
the correct dataset/weights attachments; none is configured in this local
environment. Do not put API tokens in Git, notebooks or chat.

Needed from the study team:

1. Kaggle notebook URL or owner/slug with edit permission, plus the Kaggle
   dataset slug for the primary CXR images and masks. The public original
   dataset is named in `docs/paper/short_paper.tex`, but the actual notebook
   attachment and mount path must be verified.
2. Dataset/attachment slugs for DenseNet A0/A2, ResNet A0/A3, EfficientNet
   A0/A3 and ViT A0/selected checkpoints. The supplied local ZIPs cover the
   DenseNet/EfficientNet core pairs but not the ResNet core pair or ViT.
3. A secure Kaggle CLI login/API-token setup controlled by the account owner.
   Never paste the token into a message. If a token is installed locally, it
   must stay outside this repository.
4. Study-team approval of the TV/LRG perturbation protocol and the RSNA binary
   label mapping before viewing new test/external predictions.

When connected, first verify the notebook's data paths and the fixed split
manifest. For every checkpoint: strict-load, verify class order, run a
100-image smoke test, then reproduce committed fixed-test accuracy before any
new explanation or perturbation output. The runner's `--tasks dependence`
path already enforces these gates for the CNN comparison arms. Store per-image
results and run manifests before aggregation.

For T35, run `scripts/run_t35_efficiency.py --include-vit` in one GPU notebook
session after validating the ViT architecture. The current local CPU output is
preliminary, and `fvcore`'s unsupported operations need review before GFLOPs
are reported as complete. T28/T31/T33 still require their respective data,
mapping and checkpoint checks; a successful T35 run alone does not close them.

Official Kaggle CLI documentation: `https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md`
and API authentication documentation: `https://www.kaggle.com/docs/api`.
