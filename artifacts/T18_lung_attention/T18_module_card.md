# T18 — Lung-Region Attention Module (Candidate A) — Module Card

**Owner:** Member 1 · **Bench architecture:** DenseNet121 (timm, ImageNet-pretrained) · **Status: code complete, verified locally; quantitative results pending a Kaggle GPU run (S8–S12, S14)**

Full design and step-by-step build log: `Claude Working Files/T18_Lung_Region_Attention_WBS.md`. This card summarizes what a reader (Member 2/3/5, or whoever assembles the paper) needs without reading that document end to end.

---

## 1. What this module is

Candidate A adds a lightweight, supervised spatial-attention gate between DenseNet121's feature extractor and its classification head. The gate produces a 1-channel attention map from the backbone's `[B,1024,7,7]` feature map, uses it to reweight those features (amplifying lung regions, not zeroing background), and is trained with an auxiliary loss that pushes the map to agree with the dataset's ground-truth lung mask — alongside the ordinary classification loss, not instead of it. The goal is to push the model's evidence into the lungs (measurable via faithfulness metrics) without costing classification accuracy. The module is written to be backbone-agnostic (see §7) so the same code applies to ResNet50 in T23 if Candidate A wins T22's selection.

## 2. Formal definition

Let `f ∈ ℝ^{B×C×H×W}` be the backbone feature map (`C=1024, H=W=7` for DenseNet121 at 224×224 input).

**Attention map**
```
z = W₂ · ReLU(W₁ * f + b₁) + b₂        z ∈ ℝ^{B×1×H×W}     (two 1×1 convs)
a = σ(z) ∈ (0,1)^{B×1×H×W}
```

**Gate (residual form)**
```
f' = f ⊙ (1 + a)          broadcast over channels
```
Amplifies lung evidence, never zeroes background (a plain multiplicative gate would kill gradients through low-attention cells — see the design doc §4.2 for why this form was chosen over the more obvious `f ⊙ a`).

**Head** (identical to the vanilla model's own head — no separate `nn.Linear`)
```
ŷ = Linear(GAP(f'))
```

**Loss**
```
L = CE_w(ŷ, y) + λ · L_att
L_att = BCEWithLogits(z, m̃)
m̃ = AdaptiveAvgPool2d(m, (H,W))          m ∈ {0,1}^{B×1×224×224}, m̃ ∈ [0,1]
```
`CE_w` is the team's standard class-weighted cross-entropy (`w_c = N / (K · n_c)`). Because 224/7 = 32 exactly, `m̃[i,j]` is the exact fraction of lung pixels in cell (i,j) — a soft target, not a thresholded one.

**Note for whoever reads `att_loss` in the training logs:** it does not converge to 0. BCE against a soft target has an irreducible floor equal to the target's own binary entropy (≈0.69 at the uniform-init start, ≈0.20 for a realistic lung mask at convergence) — this is expected, not a bug. See the design doc §4.4a.

## 3. Final hyperparameters

| Parameter | Value | Source |
|---|---|---|
| `reduction` (attention bottleneck) | 8 (→128 hidden channels) | design, §3.4 |
| `gate_mode` | `residual` (arm A2) | design, §4.2 |
| `target_mode` | `soft` (area-pooled mask) | design, §4.4 |
| `phase1_lr` / `phase2_lr` | 3e-4 / 3e-5 | T16's measured winning config (T13 had no result at time of writing — see §6) |
| `weight_decay` | 1e-3 | same source |
| `optimizer` / `scheduler` | AdamW / cosine | team standard |
| `unfreeze_blocks` | 1 (denseblock4 + transition3 + norm5) | team standard |
| `lambda_att` (λ*) | **PENDING** — selected by S9's sweep over {0.1, 0.3, 0.5, 1.0}, rule: largest λ within 0.5pp of arm A1's validation macro-F1, ties broken by higher validation ILAR | `artifacts/T18_lung_attention/sweep/selected_config.json` once S9 runs |

Added cost: **131,329 parameters** (+1.89% over DenseNet121's ~6.96M), **+0.0064 GFLOPs** (+0.22%) — measured for real, not estimated (§5, A6).

## 4. Comparison table

**PENDING** — produced by S11 as `artifacts/T18_lung_attention/T18_comparison_table.csv`, requires S8–S10's trained checkpoints. Expected columns once populated:

| arm | gate | λ | test_acc | test_macro_f1 | test_auc_macro | ILAR | att_dice | EIL_post | EIL_pre | Δmacro_f1 | ΔEIL_post |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 vanilla | none | 0 | | | | — | — | | (=post) | 0 (ref) | 0 (ref) |
| A1 gate-only | residual | 0 | | | | | | | | | |
| A5 CBAM (published) | — | 0 | | | | | | | | | |
| A4 guidance-only | none | λ* | | | | | | | | | |
| A2 **full** | residual | λ* | | | | | | | | | |

## 5. Acceptance criteria (design doc §1.3)

| # | Criterion | Target | Status |
|---|---|---|---|
| A1 | Attention matches lung mask | Attention-Dice ≥ 0.80 on test | **PENDING** (S11) |
| A2 | Evidence-inside-lung goes up vs. vanilla | Δ Grad-CAM EIL ≥ +0.05 (post-gate) | **PENDING** (S11) |
| A3 | Mechanism is real, not just masking | Δ Grad-CAM EIL (pre-gate) > 0 | **PENDING** (S11) |
| A4 | No accuracy tax | Δ test macro-F1 ≥ −1.0pp vs. A0 | **PENDING** (S11) |
| A5 | Ablation is complete | All required arms trained and tabulated | **PENDING** (S8–S10; code complete, not yet run) |
| A6 | Module is cheap | ≤3% params, ≤2% GFLOPs | **✅ PASS** — +1.89% params, +0.22% GFLOPs (measured, `artifacts/T18_lung_attention/acceptance_criteria_A6.json`) |
| A7 | Reproducible | seed 42, fixed split, config committed, one-command rerun | **✅ PASS** — seed 42 fixed in `configs/densenet121_lung_attention.yaml`; split manifest committed at `artifacts/splits/split_manifest_v1.csv` (verified against the real dataset, test-split per-class counts match AuxSeg's committed results exactly); the notebook runs S1→S16 top to bottom on a GPU session given the repo + dataset |
| A8 | Figures exist | ≥12 heat-map overlays + comparison grid | **PENDING** (S12; code complete, not yet run) |
| A9 | Handoff complete | per-image CSVs + module card + checkpoints delivered | **IN PROGRESS** — this document + per-image-CSV code (S11) are ready; checkpoints and CSVs themselves await the Kaggle run |

A failing verdict, once real numbers exist, is a legitimate finding to report — not a reason to keep tuning until it passes (design doc §1.3's own note).

## 6. Known limitations

- **Image-level split, not patient-level.** The COVID-19 Radiography Database ships no patient identifiers, so `docs/experiment_policy.md`'s patient-level-splitting requirement could not be satisfied. Accepted team-wide; state this in the paper's limitations section.
- **Single seed, unless S14 was run.** Check `artifacts/T18_lung_attention/runs_multiseed/multiseed_summary.json` — if its `multiseed_repeat_run` field is `false`, every number in this card and the comparison table is a single-run point estimate, not a mean over seeds. Do not imply otherwise.
- **λ\* was selected on a short training schedule** (4+6 epochs), not the full 15+40 used for final numbers — the same accepted trade-off Member 2's T16 (EfficientNet) sweep made. May be slightly off-optimal for the full schedule.
- **Lung masks are the dataset's bundled masks, not the "automatically generated" ones the submitted proposal describes** (`DNN_Project_Idea.pdf` §4.1, §5.3 say an off-the-shelf segmentation model; this implementation uses the COVID-19 Radiography Database's supplied `masks/` folder, confirmed both locally and on the Kaggle-hosted copy of the dataset). This was flagged to the team at the start of this task (design doc §12.2, gap G4) — whoever owns the Methods/Proposed Framework section needs to reconcile the wording with what was actually run.
- **7×7 attention resolution.** DenseNet121's feature map at 224×224 input is 7×7 — each attention cell covers a 32×32 pixel region. Coarser than the 224×224 mask; the soft-target area-pooling (§2) is the mitigation, not a resolution increase.
- **T13 (Member 2's DenseNet121-specific HP tuning) had no committed result at the time this module was built.** `notebooks/baseline-cnn-model-dnn-research.ipynb` only exposes `phase1_lr`/`phase2_lr` as argparse *defaults* (1e-3/1e-5), never swept. This module instead adopted the team's own committed T16 (EfficientNet-B0) sweep winner (3e-4/3e-5/wd 1e-3) as the working default — measured on this exact dataset/split/seed, both values inside the proposal's declared search space, but tuned on a different architecture. If T13 lands a DenseNet121-specific result later, `configs/densenet121_lung_attention.yaml`'s `training` block should be updated and results marked accordingly — check whether that happened before treating this card's numbers as final.
- **The submitted proposal's Table 2 (DenseNet121 baseline numbers) may itself be a placeholder** — its own caption says the values "will be populated using the Week 1 experimental results and finalized before submission." Confirm with Member 2 whether it's final before citing it as the baseline to beat.

## 7. How to reuse this module on another backbone (for T23)

`src/modules/lung_attention.py`'s `DenseNetLungAttention` and `build_model()` are backbone-agnostic despite the file's name:

1. `build_model(backbone_name="resnet50", ...)` — the model wrapper calls `timm.create_model(backbone_name, ...)` directly; no DenseNet-specific code in the forward pass.
2. ResNet50's feature map is `[B,2048,7,7]` (vs. DenseNet121's `[B,1024,7,7]`) — the module's channel count auto-derives from `backbone.num_features`, no manual change needed.
3. **The module's cost scales as `C²/reduction`.** On ResNet50 with the same `reduction=8` it's **524,801 params (+2.2% of ResNet50's ~23.5M)** — still under the A6 ≤3% target, but no longer "negligible." If T23 needs it smaller, raise `reduction` to 16 (→262K, +1.1%) — don't silently change it without noting the discrepancy from T18's own config.
4. Freeze/unfreeze: `freeze_backbone()`/`unfreeze_final_blocks()` use a small per-architecture-family registry (`_TAIL_MODULES` in the same file) already covering `densenet`/`resnet`/`efficientnet`/`vit` — T23 needs no new freeze logic, just `unfreeze_final_blocks(model, n)` as-is.
5. Everything else (`run_full_arm`, `evaluate`, `build_per_image_predictions`, the Grad-CAM harness, the comparison-table/acceptance-criteria functions) is already architecture-agnostic — pass `backbone_name="resnet50"` through and it works unchanged.
6. One thing that is NOT automatic: T23's own config file (a new `configs/resnet50_lung_attention.yaml`, copied from `configs/baseline.yaml` per repo convention, not from T18's DenseNet config) needs its own hyperparameters — don't assume T18's tuned values transfer.

## 8. Handoff

### To Member 3 (T21 ablation)
- `artifacts/T18_lung_attention/T18_comparison_table.csv` (all trained arms, once S11 runs)
- `artifacts/T18_lung_attention/runs/*/per_image_predictions.csv` (per arm)
- `artifacts/T18_lung_attention/sweep/selected_config.json` (λ* and the selection rule, for auditability)

### To Member 2 (T22 winner selection)
- A0 and A2's `EIL_post`, `EIL_pre`, and test macro-F1 from `T18_comparison_table.csv`
- **Computed using Member 5's agreed faithfulness definition** — at the time this module was built, that definition was not yet in the repo (checked `notebooks/candidate-c-grad-cam-shortcut-suppression-loss.ipynb`; it has a training-time suppression penalty, not a post-hoc scoring function). This module's own documented EIL definition (design doc §6.3: ReLU + min-max normalized Grad-CAM, fraction of mass inside the lung mask) was used instead. **Confirm with Member 5 before citing EIL numbers across candidates A/B/C** — reconcile or recompute if the definitions differ.

### To Member 5 (T28–T35, five-axis benchmark)
- A0 and A2 checkpoints (`.pt` files — **not in this PR**, see below)
- `artifacts/T18_lung_attention/T18_efficiency.csv` (real, already produced — §5, A6)
- **Three things you will hit if this isn't flagged (design doc §12.2, gaps G8/G9):**
  1. **The model returns a 3-tuple** `(logits, attention, attention_logits)`, not a plain logits tensor. Every downstream harness (calibration T28, robustness T33, efficiency T35 — though T35's own numbers are already produced above) calls `model(x)` expecting a tensor. Use `src.modules.LogitsOnly` — `LogitsOnly(model)(x)` returns just the logits. This is the single highest-value line in this handoff.
  2. **Calibration (ECE/Brier/temperature scaling):** fit temperature on the **validation** split, apply to test only. Fitting on test is exactly the model-selection-on-test `docs/experiment_policy.md` forbids.
  3. **RSNA OOD robustness (T33):** RSNA is 2-class (Pneumonia/Normal); this model's head is 4-class. A class-mapping decision is needed (e.g. Viral Pneumonia + Lung Opacity + COVID → "Pneumonia-like") — your call, but the mapping needs to be decided and documented. RSNA also ships **no lung masks**, so faithfulness can't be measured on that axis without a separate segmentation model; report accuracy/AUC drop only unless one is added.

### To myself (T23)
- `src/modules/lung_attention.py` (already backbone-agnostic — §7 above)

### Checkpoints
`.pt` files are **not** committed to this repo (per `.gitignore` and repo convention — they're large, and `artifacts/*` is checkpoint-excluded even where JSON/CSV exceptions exist). Once trained on Kaggle: upload to a Kaggle Dataset or shared Drive folder and **paste the URL here**:

- `densenet121_A0_vanilla.pt`: *(URL pending)*
- `densenet121_A2_full.pt`: *(URL pending)*
- Other arms (A1/A4/A5, optionally A3): *(URLs pending, if needed downstream)*

## 9. Contribution record (for the colour-highlighted paper, brief §11)

Everything under this heading originates from T18 (Member 1's substantive technical contribution, not writing/formatting):

- **Architecture**: the Lung-Region Attention Module itself — `LungRegionAttention`, the residual gate design, zero-init, the soft-target guidance loss (`src/modules/lung_attention.py`).
- **Novel-method comparator**: the CBAM baseline (`CBAMSpatialAttention`), added specifically to substantiate the "novel attention mechanism" claim against a published method (brief §9's "comparison with existing methods" requirement).
- **Equations**: §2 above (attention map, gate, head, loss) — reused verbatim in the paper's Methods section for Candidate A.
- **Tables**: `T18_comparison_table.csv` (§4) and the λ-sweep table (`sweep/tuning_summary.csv`).
- **Figures**: `figures/lambda_sweep.png`, `figures/attention_grid.png`, all `figures/heatmaps/*.png` (§4 of the design doc, S12).
- **Methods text**: the Candidate A subsection (draft in the design doc's Appendix B) and the ablation-design paragraph describing the 2×2 factorial (gate × supervision).
- **Infrastructure also used elsewhere**: `run_full_arm`, the freeze-registry, and the split-manifest generation (`artifacts/splits/split_manifest_v1.csv`) were built for T18 but are reused by the rest of the team's notebooks — worth noting as broader impact, not just a T18-local contribution.

Hand this list to whoever assembles the highlighted submission version.
