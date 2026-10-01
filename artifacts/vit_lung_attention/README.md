# T26: ViT-Base (avg-pool) Lung-Region Attention, seven-arm ablation

**Source:** delivered by the T26 owner on 2026-10-01 as `vit_base_lung_attention/` plus
`notebooks/archive/T_26_Vit_Base_Model_local_run_2026-10-01.ipynb`. Trained on a local Windows GPU, not on Kaggle; the predictions
carry `C:\Users\USER\Desktop\Dnn_1\...` paths. Seed 42 only.

This run **supersedes** the 2026-09-27 run (λ_att = 0.3, λ_bg = 1.0), which is kept unchanged in
`superseded_run_2026-09-27_lam0.3/` for traceability. Don't mix numbers from the two runs.

## Verification done on receipt (2026-10-01)

- **Test set:** all 7 arms predict on exactly the 3,175 test images of
  `artifacts/splits/split_manifest_v1.csv`. The notebook regenerates the split with
  `train_test_split(seed=42)` instead of loading the manifest, but the resulting test set is identical.
  The train/val membership can't be checked from the delivery, but it uses the same algorithm and seed.
- **Internal consistency:** accuracy, macro-F1, macro-AUC and the confusion matrices recompute exactly
  from `test_predictions_finetuned.csv` for every arm, and match `arm_result.json`,
  `seven_arm_summary.csv` and the notebook's printed outputs.

## Results (fine-tuned, test set, seed 42)

| Arm | λ_att | λ_bg | Acc | Macro-F1 [95% CI] | Attn-Dice | ILAR | McNemar vs A0 (Holm) |
|---|---|---|---|---|---|---|---|
| A0 vanilla (avg-pool) | – | – | 92.35% | 0.930 [0.920, 0.939] | – | – | – |
| A1 gate only | 0 | 0 | 93.86% | 0.946 [0.937, 0.954] | 0.000 | 0.184 | p = 0.001 ✱ |
| A2 full | 1.0 | 0 | 93.86% | 0.948 [0.939, 0.955] | 0.776 | 0.759 | p = 0.001 ✱ |
| A3 multiply gate | 1.0 | 0 | 93.64% | 0.943 [0.934, 0.951] | 0.775 | 0.749 | p = 0.023 ✱ |
| A4 guidance only | 1.0 | 0 | 93.07% | 0.939 [0.929, 0.947] | 0.779 | 0.759 | p = 0.315 |
| A5 CBAM | – | – | 92.63% | 0.930 [0.920, 0.940] | 0.383 | 0.240 | p = 1.000 |
| A6 full + bg loss | 1.0 | 2.0 | 93.23% | 0.941 [0.932, 0.949] | 0.759 | 0.810 | p = 0.175 |

λ_att = 1.0 and λ_bg = 2.0 were selected on validation (short schedule: 4 + 6 epochs). Sources:
`sweep/selected_config.json`, `bg_sweep/selected_bg_config.json`, `evaluation/*.csv`.

## Caveats to state wherever these numbers are used

1. **The A0 baseline stopped early.** Fine-tuning ran only 10 epochs (best epoch 5, patience 5), while
   A1/A2 ran 31 (best epoch 26). A1, the gate **without** lung supervision, gains as much as A2
   (+1.63 vs +1.77 pp macro-F1; A2 vs A1 p = 1.0). So the accuracy gain can't be attributed to lung
   supervision; it is more consistent with run-to-run variation in A0's early stop. One seed only.
2. **λ_bg = 2.0 is the edge of the sweep grid** {0, 0.1, 0.3, 0.5, 1.0, 2.0}, so the optimum may lie
   beyond it. Validation F1 was flat across the grid (0.9271–0.9275).
3. **No significance test on evidence location yet.** The McNemar column tests accuracy only. The
   cross-backbone winner rule also needs Wilcoxon on per-image Grad-CAM EIL, which doesn't exist yet.
   **No ViT winner is selected.**

## Missing from the delivery (needed before T26 can close)

| Item | Why it's needed | Status |
|---|---|---|
| Model checkpoints `seven_arm/<arm>/final_model.pt` (7 × ~340 MB) | Every T28–T35 step (calibration, Grad-CAM, perturbation, RSNA, efficiency) | **Missing.** Saved on the owner's PC, not sent |
| Per-image Grad-CAM EIL (post- and pre-gate) on the 1,000-image CAM subset | Winner rule (Wilcoxon), cross-backbone EIL table | **Not computed.** The notebook has no Grad-CAM cells |
| Per-image ILAR | Paired statistics on attention | Only arm means delivered |
| `winner_selection.json`, `significance_tests_vs_A0.json` (McNemar + Wilcoxon) | Same rule as T23/T25 | Missing (needs EIL) |
| 3-seed repeat (123, 2026) of A0 + selected winner | Caveat 1 above; paper-grade claim | Missing |
| Efficiency (GFLOPs, latency) | T35 | Missing (params only) |
| Module card | Same as T18/T23 | Missing |

## Derived files (added on receipt; delivered files are unchanged)

`scripts/build_t26_vit_tables.py` generates:
- `seven_arm/<arm>/per_image_predictions.csv`: team schema (relative image paths; `ilar`/`eil_*` = NaN)
- `T26_comparison_table.csv`: same columns as `T23_comparison_table.csv`; EIL columns NaN

## Replacement run

`notebooks/T_26_Vit_Base_Model.ipynb` re-runs T26 on Kaggle with the shared `src/modules`
pipeline (same protocol and outputs as T23). It fixes the gaps listed above: checkpoints with their
config, per-image Grad-CAM EIL, the full winner rule, a 3-seed repeat, efficiency and a module card.
Its results go to `artifacts/T26_vit_base_lung_attention/` and will supersede this folder.
