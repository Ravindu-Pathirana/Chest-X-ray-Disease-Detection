# T18 — Lung-Region Attention Module (Candidate A) — Module Card

**Owner:** Member 1 · **Bench architecture:** DenseNet121 (timm, ImageNet-pretrained) · **Status: trained end-to-end on Kaggle (2026-09-09/10) — acceptance criteria A1–A4, A6, A7, A8 PASS. Checkpoints/comparison table/figures are still only inside the Kaggle session and need S15's commit to land in this repo — treat the numbers below as real but the linked files as pending.**

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
| `phase1_lr` / `phase2_lr` | 4.12e-3 / 6.2e-5 | T13's measured winning config (landed after this card was first written — see §6) |
| `weight_decay` | 1.87e-5 | same source |
| `optimizer` / `scheduler` | Adam / plateau | same source (was AdamW / cosine team-standard defaults, superseded — see §6) |
| `unfreeze_blocks` | 3 (denseblock4/3/2 + transition2/3 + norm5) | same source (was 1) |
| `batch_size` | 64 | same source (was 32) |
| `drop_rate` (timm head dropout) | 0.4303 | same source — new knob, added to `build_model`/`DenseNetLungAttention` specifically to carry this value (§4.3: must match Member 2's baseline head exactly) |
| `lambda_att` (λ*) | **1.0** — selected by S9's sweep over {0.0, 0.1, 0.3, 0.5, 1.0}: largest λ whose validation macro-F1 stayed within 0.5pp of the λ=0 reference (0.9411), ties broken by higher validation ILAR. 1.0 won outright (val macro-F1 0.9504, well inside tolerance) | `artifacts/T18_lung_attention/sweep/selected_config.json` (pending S15 commit — currently only in the Kaggle session) |

Added cost: **131,329 parameters** (+1.89% over DenseNet121's ~6.96M), **+0.0064 GFLOPs** (+0.22%) — measured for real, not estimated (§5, A6).

## 4. Comparison table

Real, single-seed (42) results from the 2026-09-09 Kaggle run. Full detail (per-image predictions, confusion matrices) lives in `artifacts/T18_lung_attention/T18_comparison_table.csv` once S15 commits it — currently only inside the Kaggle session's `/kaggle/working/`. EIL was computed for the 4 arms S11 actually scored (A0, A1, A4, A2); A5 and A3 don't have EIL numbers yet.

| arm | gate | λ | test_acc | test_macro_f1 | test_auc_macro | ILAR | att_dice | att_iou | EIL_post | EIL_pre | Δmacro_f1 | ΔEIL_post |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 vanilla | none | 0 | 0.9468 | 0.9511 | 0.9932 | — | — | — | 0.2962 | (=post) | 0 (ref) | 0 (ref) |
| A1 gate-only | residual | 0 | 0.9452 | 0.9497 | 0.9922 | 0.2440 | 0.0803 | 0.0495 | 0.2841 | 0.2839 | −0.0014 | −0.0121 |
| A5 CBAM (published) | multiply | 0 | 0.9439 | 0.9476 | 0.9922 | 0.2725 | 0.4152 | 0.2654 | — | — | −0.0035 | — |
| A4 guidance-only | none | 1.0 | 0.9461 | 0.9499 | 0.9932 | 0.6313 | 0.8398 | 0.7293 | 0.3172 | (=post) | −0.0012 | +0.0210 |
| A2 **full** | residual | 1.0 | 0.9465 | 0.9508 | 0.9929 | 0.6253 | 0.8390 | 0.7285 | 0.3735 | 0.3111 | **−0.0003** | **+0.0773** |
| A3 multiply | multiply | 1.0 | 0.9424 | 0.9446 | 0.9927 | 0.6230 | 0.8402 | 0.7303 | — | — | −0.0065 | — |

**3-seed repeat (S14, seeds 42/123/2026), A0 vs. A2 only:**

| arm | test_macro_f1 (mean ± std) | EIL_post (mean ± std) |
|---|---|---|
| A0 vanilla | 0.9512 ± 0.0007 | 0.2996 ± 0.0062 |
| A2 full | 0.9504 ± 0.0045 | 0.3701 ± 0.0060 |

Headline: **Δmacro-F1 = −0.0008, ΔEIL_post = +0.0705 ± 0.0086** across seeds — the single-seed numbers above aren't a fluke.

## 5. Acceptance criteria (design doc §1.3)

| # | Criterion | Target | Status |
|---|---|---|---|
| A1 | Attention matches lung mask | Attention-Dice ≥ 0.80 on test | **✅ PASS** — 0.8390 (A2 vs. target ≥0.80) |
| A2 | Evidence-inside-lung goes up vs. vanilla | Δ Grad-CAM EIL ≥ +0.05 (post-gate) | **✅ PASS** — +0.0773 single-seed, +0.0705 ± 0.0086 across 3 seeds |
| A3 | Mechanism is real, not just masking | Δ Grad-CAM EIL (pre-gate) > 0 | **✅ PASS** — +0.0150 |
| A4 | No accuracy tax | Δ test macro-F1 ≥ −1.0pp vs. A0 | **✅ PASS** — −0.03pp single-seed, −0.08pp 3-seed mean |
| A5 | Ablation is complete | All required arms trained and tabulated | **✅ PASS** — all 6 arms (A0–A5) trained with checkpoints and test numbers (§4) |
| A6 | Module is cheap | ≤3% params, ≤2% GFLOPs | **✅ PASS** — +1.89% params, +0.22% GFLOPs (measured, `artifacts/T18_lung_attention/acceptance_criteria_A6.json`). Note: re-running the S13 efficiency cell on a fresh Kaggle session without `pip install fvcore` returns GFLOPs=NaN and a false FAIL — this committed result (computed with fvcore available) is the trustworthy one. |
| A7 | Reproducible | seed 42, fixed split, config committed, one-command rerun | **✅ PASS** — seed 42 fixed in `configs/densenet121_lung_attention.yaml`; split manifest committed at `artifacts/splits/split_manifest_v1.csv` (verified against the real dataset, test-split per-class counts match AuxSeg's committed results exactly, and now also M5's independent P02 mask-verification pass); the notebook ran S1→S14 top to bottom on a Kaggle GPU session against the real repo + dataset |
| A8 | Figures exist | ≥12 heat-map overlays + comparison grid | **✅ PASS** — 12 heat-map overlays + `attention_grid.png` + `lambda_sweep.png` generated (pending S15 commit into this repo) |
| A9 | Handoff complete | per-image CSVs + module card + checkpoints delivered | **IN PROGRESS** — this card is now updated with real numbers; per-image CSVs, the comparison table, figures and checkpoints still need S15's commit (currently only in the Kaggle session) |

A failing verdict, once real numbers exist, is a legitimate finding to report — not a reason to keep tuning until it passes (design doc §1.3's own note).

## 6. Known limitations

- **Image-level split, not patient-level.** The COVID-19 Radiography Database ships no patient identifiers, so `docs/experiment_policy.md`'s patient-level-splitting requirement could not be satisfied. Accepted team-wide; state this in the paper's limitations section.
- **Multi-seed repeat (S14) was actually run** — 3 seeds (42, 123, 2026) for A0 and A2, real mean±std in §4 above. **Caveat:** the notebook's cell 72 (an "alternative DoD" fallback meant only for when S14 is skipped) wrote to the same `runs_multiseed/multiseed_summary.json` path as cell 70 and ran unconditionally after it, silently overwriting the real 3-seed data with a false "single seed only, S14 not run" statement. This has been fixed in the notebook (cell 72 is now guarded to skip itself when a real multiseed run already happened) — but if you're looking at a `multiseed_summary.json` from before 2026-09-10, verify its `multiseed_repeat_run` field isn't `false` before trusting it; regenerate from cell 70 if it is. The real numbers are the ones in §4 above, taken directly from cell 70's printed output.
- **λ\* was selected on a short training schedule** (4+6 epochs), not the full 15+40 used for final numbers — the same accepted trade-off Member 2's T16 (EfficientNet) sweep made. May be slightly off-optimal for the full schedule.
- **Lung masks are the dataset's bundled masks, not the "automatically generated" ones the submitted proposal describes** (`DNN_Project_Idea.pdf` §4.1, §5.3 say an off-the-shelf segmentation model; this implementation uses the COVID-19 Radiography Database's supplied `masks/` folder, confirmed both locally and on the Kaggle-hosted copy of the dataset, and now independently re-verified by M5's P02 mask-quality audit — `artifacts/segmentation/mask_quality_report.csv`, all 21,165 images have a matching valid mask). This was flagged to the team at the start of this task (design doc §12.2, gap G4) — whoever owns the Methods/Proposed Framework section needs to reconcile the wording with what was actually run.
- **7×7 attention resolution.** DenseNet121's feature map at 224×224 input is 7×7 — each attention cell covers a 32×32 pixel region. Coarser than the 224×224 mask; the soft-target area-pooling (§2) is the mitigation, not a resolution increase.
- **T13 (Member 2's DenseNet121-specific HP tuning) landed before this run** (`notebooks/DenseNet_HPO_train.ipynb`, a 20-trial Optuna search retrained at full budget: 94.83% test accuracy, macro-F1 0.9505) and its config was confirmed working end-to-end in this Kaggle run (phase1_lr 4.12e-3, phase2_lr 6.2e-5, Adam, plateau scheduler, unfreeze_blocks=3, batch_size=64, drop_rate=0.4303 — all matching §3 above). Note T13's HPO computed its own stratified split (`stratified_split`/`load_base_dataset_and_split` in that notebook) rather than loading the committed `artifacts/splits/split_manifest_v1.csv`; same seed and stratification method, so it matches, but this wasn't independently re-verified beyond the split manifest's own checks (§5, A7).
- **S9's sweep config still labels its provenance as "T16 winner D_more_wd (T13 unavailable at time of writing)"** in `sweep/selected_config.json` — that string is stale (T13's config was in fact used, per the point above); it's a cosmetic leftover in the sweep code, not a sign the wrong hyperparameters were used. Worth a one-line fix next time S9 is re-run, not urgent.
- **A5 (CBAM) and A3 (multiply-gate) don't have Grad-CAM EIL numbers** — S11's `arms_for_table` loop only scored A0/A1/A4/A2 (see §4); if a complete 6-arm EIL comparison is needed for the paper, extend that loop to A5 and A3 and re-run S11 (no retraining required, checkpoints already exist).
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

## 10. Optional extensions (opt-in — do not affect the frozen A0-A5 ablation)

Added after an external novelty/literature review of T18: the core critique was
that lung-mask-supervised attention is no longer novel by itself, and that "lung
region" ≠ "disease evidence" — attention-Dice/ILAR only measure where the
attention map *points*, not whether the classifier's decision still depends on
background content. A full architectural redesign (dual lung+disease attention,
class-specific attention, adversarial CAM alignment) was deliberately rejected
as too risky this close to T18's first real Kaggle run and out of step with this
project's own framing as a five-axis *comparison*, not a novel-architecture
paper. These two additions were judged worth the (small) cost instead:

- **Background-suppression loss** (`src/modules/lung_attention.py::background_suppression_loss`,
  `module.lambda_bg` in the config, default `0.0`): directly penalizes attention
  MASS placed outside the lung mask, complementing `attention_guidance_loss`'s
  BCE-toward-the-mask term. Every A0-A5 arm keeps `lambda_bg=0.0` — this is a
  separate, later "A2+bg" experiment, not a redefinition of A2. Fully wired
  through `run_epoch`/`train_phase`/`run_full_arm` and logged as
  `train_bg_loss`/`val_bg_loss` regardless of `lambda_bg`'s value (same
  diagnostic-even-when-off convention as `att_loss`).
- **Counterfactual background-perturbation robustness** (`src/modules/counterfactual.py`):
  pure post-hoc evaluation on an already-trained checkpoint — zeroes,
  randomizes, or adds noise to the background only (lung pixels held
  pixel-identical) and measures how much the predicted distribution shifts
  (`counterfactual_stability`) and how often the predicted class flips. Unlike
  every other metric in this card, this one is model-architecture-agnostic
  (works on arm A0 exactly the same way) and directly tests shortcut
  *reliance*, not attention *placement* — the strongest single piece of
  evidence this project's "trustworthy CXR" framing can produce. Not yet run
  against real checkpoints (none exist yet — see this card's header); intended
  as an S11-adjacent step once A0/A2 checkpoints exist. Suggested notebook
  cell, reusing the existing `ARM_CHECKPOINTS`/`load_arm_model` helpers from S11:

  ```python
  from src.modules import LogitsOnly, evaluate_counterfactual_robustness

  CF_CSV = REPO_ROOT / "artifacts" / "T18_lung_attention" / "counterfactual_robustness.csv"
  for arm_name, ckpt_path in ARM_CHECKPOINTS.items():
      model, _cfg, _classes = load_arm_model(ckpt_path)
      evaluate_counterfactual_robustness(
          LogitsOnly(model), test_loader, device=device,
          modes=("zero", "shuffle", "noise"), arm_name=arm_name, output_csv=CF_CSV,
      )
  ```

Both additions have unit tests (`tests/test_lung_attention.py`,
`tests/test_counterfactual.py`) using toy/synthetic data verified against
hand-computed expectations, but neither has been exercised against a real
checkpoint or the real dataset — treat the numbers as implemented-and-tested,
not yet validated at scale.
