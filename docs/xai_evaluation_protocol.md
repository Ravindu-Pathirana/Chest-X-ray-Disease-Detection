# Explainability evaluation protocol — draft

Status: **draft for future inference**, started on `explainable-AI` on 30
September 2026. The analyses of previously saved test outputs are
retrospective. This document does not claim those decisions were registered
before the original results were viewed.

## Research questions and evidence levels

1. **Prediction quality:** compare accuracy and macro-F1 on the fixed test
   split. This is already available for the three CNNs.
2. **Anatomical localization:** compare predicted-class Grad-CAM energy inside
   the lung mask (EIL) on identical images. The three-CNN, three-seed results
   are available. This says *where* heatmap energy lies, not *what* feature
   drove the decision.
3. **Input dependence:** perturb lung and background regions and measure how
   the full predicted-class distribution changes, with matched interventions
   and per-image paired outputs. This is still missing for the cross-CNN study.
4. **External transfer:** evaluate the frozen classifier on a separately
   prepared external dataset under a predeclared label mapping. No RSNA
   performance claim is currently supported by repository outputs.

## Locked properties of the existing CNN localization analysis

- Primary data split: `artifacts/splits/split_manifest_v1.csv` (3,175 test
  images). Test labels are not used for model selection.
- EIL is `src.modules.attention_metrics.energy_inside_lung`: fraction of
  nonnegative Grad-CAM mass inside the resized binary lung mask. The CAM
  targets the model's predicted class. CAMs are computed in fp32.
- The committed fixed 1,000-image CAM subset is identical for DenseNet121,
  ResNet50 and EfficientNet-B0, and baseline/selected arms are paired by
  `image_path` within each seed.
- The selected comparison arms in the committed CNN closeout are DenseNet
  `A2_full`, ResNet `A3_multiply` and EfficientNet `A3`, each against A0.
  This protocol does not retrospectively change arm selection.
- Missing measurements remain missing, not zero. Every reported aggregate
  must trace to a source per-image file.

## Proposed analysis of saved results

`scripts/build_explainability_baseline.py` computes selected-minus-baseline
EIL differences on the same 1,000 images, with a paired-image percentile
bootstrap (2,000 resamples, RNG seed 42) and a two-sided Wilcoxon test.
P-values are Holm-adjusted across three backbones within each seed. This is
a retrospective precision analysis of committed results, not new evidence
of causal explanation faithfulness.

## Checkpoint readiness from supplied archives

The four supplied ZIP archives were inspected without extracting or committing
weights. `scripts/audit_checkpoint_archives.py --strict-load` verified the
checkpoint state dictionaries against their declared architectures. Its
inventory and coverage report are in `artifacts/explainable_ai/`. There are 22
strict-loadable final checkpoints, covering 12 of the 18 required core runs:
DenseNet A0/A2 and EfficientNet A0/A3 at seeds 42, 123 and 2026. The six
ResNet50 A0/A3 checkpoints are absent, including from the archive named
`resnet (1).zip`; that archive contains a DenseNet A2 checkpoint instead. A
non-final smoke checkpoint was excluded from coverage.

This is a *model-load* check, not an inference smoke test. The archives do not
provide the primary test image directory or lung-mask images. No new CAM,
pointing-game, perturbation, or counterfactual result can be generated from
these archives alone. The supplied weights also remain outside Git because
they are large binary research artifacts.

To recheck a supplied ZIP, run
`python scripts/audit_checkpoint_archives.py --archive <zip-path> --strict-load`.
Pass `--archive` once per ZIP; the script writes only the small inventory and
coverage report, without extracting checkpoints into the repository.

## Decisions to freeze before **new** inference

| Decision | Working proposal | Status |
|---|---|---|
| Primary input-dependence metric | For each image and intervention, total-variation distance `dP = 0.5 × sum_c abs(p_original,c - p_perturbed,c)`. Per-image `LRG = dP_lung - dP_background`, calculated separately for blur and swap. Original-predicted-class probability drop is secondary, not LRG. | Proposed; team sign-off pending |
| Intervention realism | Lung and background blur and different-class donor swap are the primary matched modes. A randomly selected background region with the same pixel count as the lung mask is the area control for each mode; zero/shuffle/noise remain stress tests. If no equal-sized background region exists, stop and agree an exclusion rule before inference. | Code implemented; inference pending |
| Perturbation randomness | Use seed 42; freeze `artifacts/explainable_ai/swap_pairs_seed42.csv` from the fixed test manifest. Reuse the same donor pair and area-matched pixel selection for every arm/backbone. | Pair generation implemented; team sign-off pending |
| Multiple comparisons | Define comparison family before testing new arms; report adjusted p-values with CIs and effect sizes | Pending |
| RSNA binary mapping | Specify which primary classes are scored positive before external predictions are viewed | Team decision required |
| ViT explanation target | Audit average-pool/patch-token path and comparability before inclusion in a CNN-versus-ViT claim | Pending |
| Qualitative examples | Freeze image identifiers and selection categories before producing new panels | Pending |

The legacy `run_cnn_closeout_inference.py` still covers zero/blur lung
occlusion and zero/shuffle/noise background counterfactuals. Its probability
drop output is not TV-based LRG. The new `xai_dependence.py` path adds matched
blur/swap and area controls without changing those legacy artifacts. Before
new inference, verify the checkpoint against fixed-test accuracy and review
this protocol with the study team. Existing results remain retrospective;
this amendment is not a claim of preregistration for them.

The donor manifest can be regenerated without image files using
`python scripts/build_xai_swap_pairs.py`. Once images, masks and a verified
checkpoint are available, use `scripts/run_cnn_closeout_inference.py` with
`--tasks dependence`, the correct `--backbone`, `--arm`, `--seed`,
`--checkpoint`, `--data-dir` and a dedicated `--output-dir`. That task runs
the 100-image smoke test, compares full-test accuracy with the committed
per-image predictions (tolerance 0.0001), and stops on a mismatch before
writing new dependence results. It then writes `dependence_per_image.csv`
and `dependence_summary.csv`; no result is claimed until that run succeeds.

## Claim boundary

Use **anatomical localization** for current EIL and attention-map results.
Do not infer reduced shortcut reliance or clinical relevance from heatmap
location alone. External evaluation cannot demonstrate transfer to all
hospitals from one RSNA dataset.
