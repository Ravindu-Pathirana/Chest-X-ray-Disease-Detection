# T26: ViT-Base (avg-pool) Lung-Region Attention, local seven-arm run

**What this folder holds:** the T26 owner's local run of 2026-10-01 (PR #38), under
`runs/vit_base_lung_attention/`. Trained on a local Windows GPU, not on Kaggle; file paths inside the
CSVs are `C:\Users\USER\Desktop\Dnn_1\...`. Seed 42 only. The notebook that produced the training
results is kept as `notebooks/archive/T_26_Vit_Base_Model_local_run_2026-10-01.ipynb`. The Grad-CAM
code that produced `gradcam_*` is not in the repo.

It replaced the 2026-09-27 run (λ_att = 0.3, λ_bg = 1.0), which PR #38 deleted; that run is in git
history only. Don't mix numbers from the two.

## Verification done on receipt

- **Test set:** all 7 arms predict on exactly the 3,175 test images of
  `artifacts/splits/split_manifest_v1.csv`. The notebook regenerates the split with
  `train_test_split(seed=42)` instead of loading the manifest, but the resulting test set is identical.
  Train/val membership can't be checked from the delivery; it uses the same algorithm and seed.
- **Internal consistency:** accuracy, macro-F1, macro-AUC and the confusion matrices recompute exactly
  from `test_predictions_finetuned.csv` for every arm, and match `arm_result.json` and
  `seven_arm_summary.csv`.

## Results (fine-tuned, test set, seed 42)

| Arm | λ_att | λ_bg | Acc | Macro-F1 [95% CI] | Attn-Dice | EIL pre → post (n = 200) | McNemar vs A0 (Holm) |
|---|---|---|---|---|---|---|---|
| A0 vanilla (avg-pool) | – | – | 92.35% | 0.930 [0.920, 0.939] | – | 0.236 → 0.236 | – |
| A1 gate only | 0 | 0 | 93.86% | 0.946 [0.937, 0.954] | 0.000 | 0.240 → 0.240 | p = 0.001 ✱ |
| A2 full | 1.0 | 0 | 93.86% | 0.948 [0.939, 0.955] | 0.776 | 0.265 → 0.340 | p = 0.001 ✱ |
| A3 multiply gate | 1.0 | 0 | 93.64% | 0.943 [0.934, 0.951] | 0.775 | 0.411 → 0.773 | p = 0.023 ✱ |
| A4 guidance only | 1.0 | 0 | 93.07% | 0.939 [0.929, 0.947] | 0.779 | 0.230 → 0.230 | p = 0.315 |
| A5 CBAM | – | – | 92.63% | 0.930 [0.920, 0.940] | 0.383 | 0.229 → 0.229 | p = 1.000 |
| A6 full + bg loss | 1.0 | 2.0 | 93.23% | 0.941 [0.932, 0.949] | 0.759 | 0.264 → 0.336 | p = 0.175 |

Lung area is 0.235 of the image on the EIL subsample, so an EIL of 0.236 (A0) is what a uniform map
would score. Sources: `runs/vit_base_lung_attention/{seven_arm,sweep,bg_sweep,evaluation}/`.

## Caveats to state wherever these numbers are used

1. **The A0 baseline stopped early.** Fine-tuning ran only 10 epochs (best epoch 5, patience 5 on
   validation macro-F1), while A1/A2 ran 31 (best epoch 26). A1, the gate **without** lung
   supervision, gains as much accuracy as A2 (+1.63 vs +1.77 pp macro-F1; A2 vs A1 p = 1.0), and at
   this run's attention initialisation (≈0.0025) A1 is almost the same network as A0. So the accuracy
   gain can't be attributed to lung supervision; it is more consistent with run-to-run variation. One
   seed only.
2. **EIL is on 200 images, with the owner's own Grad-CAM code.** The CNN backbones use 1,000 images and
   `src/modules/gradcam.py`. Treat the ViT EIL as indicative, not as a row in the cross-backbone table.
3. **λ_bg = 2.0 is the edge of the sweep grid** {0, 0.1, 0.3, 0.5, 1.0, 2.0}.
4. **Protocol differs from T18/T23:** batch 8, early stopping on validation macro-F1, attention metrics
   computed at 14×14, CBAM with a channel branch, test evaluation under mixed precision.
5. **No winner is selected.** There is no Wilcoxon test on paired EIL and no `winner_selection.json`.

## Still missing from this run

| Item | Needed for | Status |
|---|---|---|
| Model checkpoints `seven_arm/<arm>/final_model.pt` (7 × ~340 MB) | T28–T35 | Not delivered |
| Winner selection (McNemar + Wilcoxon on EIL), 3-seed repeat | T26 close-out | Not done |
| Efficiency (GFLOPs, latency), module card | T35, hand-off | Not done |

## Replacement run (done)

T26 was re-run on Kaggle on 2026-10-02 with `notebooks/T_26_Vit_Base_Model.ipynb`: the shared
`src/modules` pipeline, the same protocol and the same outputs as T23. Its results are in
`artifacts/T26_vit_base_lung_attention/` and **supersede this folder** for every reported ViT number:
winner A3_multiply, confirmed over 3 seeds; A2 passes all four acceptance criteria. See
`artifacts/T26_vit_base_lung_attention/T26_module_card.md`. This folder is kept as the record of the
earlier local run.

## Derived files (not part of the delivery)

`scripts/build_t26_vit_tables.py` generates, without changing any delivered file:
- `runs/vit_base_lung_attention/seven_arm/<arm>/per_image_predictions.csv`: the team's per-image
  schema, relative image paths, `eil_post`/`eil_pre` for the 200 Grad-CAM images and NaN elsewhere.
- `T26_comparison_table.csv`: same columns as `T23_comparison_table.csv`, plus `eil_n_images`.
