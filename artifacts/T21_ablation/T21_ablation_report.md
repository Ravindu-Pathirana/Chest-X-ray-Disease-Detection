# T21 - DenseNet121 Candidate Ablation

**Task:** Ablate each shortcut-suppression candidate against vanilla DenseNet121.

**Status:** Complete from committed artifact CSV/JSON outputs.

**Bench architecture:** DenseNet121

**Test set:** 3,175 images, fixed class order:

| Class | Support |
|---|---:|
| COVID | 542 |
| Lung_Opacity | 902 |
| Normal | 1,529 |
| Viral Pneumonia | 202 |

All four confusion matrices use the same class order and same row supports, so the reported runs are directly comparable at the test-set level.

## Source Artifacts

| Arm | Task | Source file |
|---|---|---|
| DenseNet121 vanilla | T12/T18 reference arm | `artifacts/T18_lung_attention/runs/A0_vanilla/phase2_finetune_summary_metrics.csv` |
| Candidate A - Lung-Region Attention | T18 | `artifacts/T18_lung_attention/runs/A2_full/phase2_finetune_summary_metrics.csv` |
| Candidate B - Auxiliary Segmentation Head | T19 | `artifacts/densenet121_auxseg/runs/densenet121_auxseg_tuned_lam0_1/phase2_finetune_summary_metrics.csv` |
| Candidate C - Grad-CAM Shortcut-Suppression Loss | T20 | `artifacts/candidate_c_suppression/runs/densenet121_candidate_c/candidate_c_test_results_summary_metrics.csv` |

## Ablation Table

| Model | Accuracy | Macro F1 | Macro Recall | Macro Precision | Macro ROC-AUC | Delta Macro F1 vs vanilla | Delta Accuracy vs vanilla |
|---|---:|---:|---:|---:|---:|---:|---:|
| DenseNet121 vanilla | 0.9468 | 0.9511 | 0.9487 | 0.9540 | 0.9932 | 0.0000 | 0.0000 |
| Candidate A - Lung-Region Attention | 0.9465 | 0.9508 | 0.9466 | 0.9560 | 0.9929 | -0.0003 | -0.0003 |
| Candidate B - Auxiliary Segmentation Head tuned | 0.9436 | 0.9493 | 0.9412 | 0.9584 | 0.9920 | -0.0018 | -0.0031 |
| Candidate C - Grad-CAM Shortcut-Suppression Loss | 0.9553 | 0.9594 | 0.9605 | 0.9585 | 0.9946 | +0.0083 | +0.0085 |

## Confusion Matrix Sanity Check

| Model | COVID | Lung_Opacity | Normal | Viral Pneumonia | Total |
|---|---:|---:|---:|---:|---:|
| DenseNet121 vanilla | 542 | 902 | 1,529 | 202 | 3,175 |
| Candidate A - Lung-Region Attention | 542 | 902 | 1,529 | 202 | 3,175 |
| Candidate B - Auxiliary Segmentation Head | 542 | 902 | 1,529 | 202 | 3,175 |
| Candidate C - Grad-CAM Shortcut-Suppression Loss | 542 | 902 | 1,529 | 202 | 3,175 |

## Interpretation

Candidate C gives the strongest classification performance in this ablation. It improves over vanilla DenseNet121 by:

- **+0.85 percentage points accuracy**
- **+0.83 percentage points macro F1**
- **+0.14 percentage points macro ROC-AUC**

Candidate A preserves vanilla classification performance almost exactly while improving the T18 faithfulness signal reported in `artifacts/T18_lung_attention/T18_comparison_table.csv`. Its macro F1 drop is negligible at -0.03 percentage points.

The tuned Candidate B run fixes most of the earlier classification drop while improving segmentation quality. It remains slightly below vanilla DenseNet121 on macro F1 (-0.18 percentage points) and accuracy (-0.31 percentage points), but the drop is now small enough that AuxSeg can be treated as a viable candidate if its segmentation/faithfulness benefit is valued.


## Candidate B Audit Note

Candidate B's original low classification score appears to be a real output of the first committed run, not a corrupted CSV:

- `phase2_finetune_summary_metrics.csv`, `phase2_finetune_test_results.json`, and `phase2_finetune_confusion_matrix.csv` agree.
- Its confusion matrix uses the same class order and same test support as vanilla, Candidate A, and Candidate C.
- The training history supports the final score: Candidate B reaches only about 0.897 validation accuracy after 36 phase-2 epochs, then early-stops.

However, Candidate B is not tuned as strongly as the later DenseNet121 runs. The AuxSeg notebook uses these defaults:

| Setting | Candidate B AuxSeg | T13/T18 tuned DenseNet setting |
|---|---:|---:|
| `phase1_lr` | 0.001 | 0.00412 |
| `phase2_lr` | 0.00001 | 0.000062 |
| `unfreeze_blocks` | 1 | 3 |
| `batch_size` | 32 | 64 |
| `optimizer` | AdamW | Adam |
| `weight_decay` | 0.0001 | 0.0000187 |
| `scheduler` | cosine | plateau |
| checkpoint selection | total multitask validation loss | validation loss in tuned DenseNet protocol |
| `lambda_seg` | 0.3 | not swept in committed artifacts |

A tuned rerun was added at `artifacts/densenet121_auxseg/runs/densenet121_auxseg_tuned_lam0_1/` using the stronger DenseNet121 settings and `lambda_seg=0.1`. This raises Candidate B to accuracy 0.9436, macro F1 0.9493, macro ROC-AUC 0.9920, segmentation Dice 0.9397, and segmentation IoU 0.8877. T21 therefore reports the tuned AuxSeg result in the main ablation table, while keeping this note to explain why the first AuxSeg row was not used as the final Candidate B comparison.

## T22 Handoff Notes

T22 should not select the winner from classification metrics alone. The project goal is improved trustworthiness without meaningful classification loss. Based on this T21 table:

- Candidate C is the classification winner.
- Candidate A is the strongest "no accuracy loss plus explicit lung-faithfulness gain" candidate from the available T18 metrics.
- Tuned Candidate B is now close to vanilla classification performance and has the strongest explicit segmentation output, but still trails Candidate C and vanilla on macro F1.

Before final T22 winner selection, compare the available faithfulness/robustness evidence for Candidate A and Candidate C using the same definition wherever possible. Candidate A already reports Grad-CAM energy-inside-lung values in `artifacts/T18_lung_attention/T18_comparison_table.csv`; Candidate C should ideally have the same faithfulness metric or a clearly documented equivalent.

