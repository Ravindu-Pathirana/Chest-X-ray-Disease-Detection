Yes. Now that we have confirmed the primary dataset and that it already contains lung masks, we can define a complete end-to-end pipeline that matches your actual proposal, rather than a generic chest X-ray project.

Your proposal defines the overall methodology as: primary COVID-19 Radiography data + external RSNA data → lung-region information → shared preprocessing and fixed stratified split → four vanilla architectures → shortcut-suppression module development → winning module across all architectures → five-axis trustworthiness benchmark. CS3631_Project_Proposal_Group04.pdf The shorter proposal confirms 224×224 inputs, ImageNet normalization, class-weighted cross-entropy, two-phase transfer learning, early stopping, and identical experimental conditions across architectures. DNN.pdf

Because the primary dataset already provides lung masks, I would implement the project through the following 17 processes.

⸻

Complete project pipeline

PRIMARY DATA
COVID-19 Radiography Database
21,165 CXR images + supplied lung masks
│
▼
┌──────────────────────────────────────┐
│ P01 — Dataset Audit │
└──────────────────┬───────────────────┘
▼
┌──────────────────────────────────────┐
│ P02 — Image ↔ Mask Verification │
└──────────────────┬───────────────────┘
▼
┌──────────────────────────────────────┐
│ P03 — Master Dataset Manifest │
└──────────────────┬───────────────────┘
▼
┌──────────────────────────────────────┐
│ P04 — Fixed 70/15/15 Split │
└──────────────────┬───────────────────┘
▼
┌──────────────────────────────────────┐
│ P05 — Preprocessing & Augmentation │
└──────────────────┬───────────────────┘
▼
┌──────────────────────────────────────┐
│ P06 — Dataset + DataLoaders │
└──────────────────┬───────────────────┘
▼
┌──────────────────────────────────────┐
│ P07 — Class-Imbalance Handling │
└──────────────────┬───────────────────┘
▼
┌──────────────────────────────────────┐
│ P08 — Stage 1 Vanilla Models │
└──────────────────┬───────────────────┘
▼
┌──────────────────────────────────────┐
│ P09 — Transfer Learning + HP Tuning │
└──────────────────┬───────────────────┘
▼
┌──────────────────────────────────────┐
│ P10 — Internal Baseline Evaluation │
└──────────────────┬───────────────────┘
▼
┌──────────────────────────────────────┐
│ P11 — Shortcut Module Development │
└──────────────────┬───────────────────┘
▼
┌──────────────────────────────────────┐
│ P12 — Ablation + Winner Selection │
└──────────────────┬───────────────────┘
▼
┌──────────────────────────────────────┐
│ P13 — Cross-Architecture Testing │
└──────────────────┬───────────────────┘
▼
┌──────────────────────────────────────┐
│ P14 — Five-Axis Benchmark │
└──────────────────┬───────────────────┘
▼
┌──────────────────────────────────────┐
│ P15 — RSNA External Evaluation │
└──────────────────┬───────────────────┘
▼
┌──────────────────────────────────────┐
│ P16 — Statistical/Comparative │
│ Analysis │
└──────────────────┬───────────────────┘
▼
┌──────────────────────────────────────┐
│ P17 — Final Results + Paper │
└──────────────────────────────────────┘

⸻

P01 — Dataset audit

Objective

Before changing the images or training a model, establish exactly what data you have.

Your primary dataset has four classes:

COVID-19
Normal
Lung Opacity
Viral Pneumonia

The proposal specifically notes that these classes are imbalanced. DNN.pdf

Step 1 — Locate all images

Scan the four class directories and identify every CXR image.

Create records such as:

COVID-1.png → COVID
Normal-1.png → Normal
Lung_Opacity-1 → Lung Opacity
Viral-1.png → Viral Pneumonia

Step 2 — Count images

Produce a table:

Class Number
COVID-19 count
Normal count
Lung Opacity count
Viral Pneumonia count
Total count

This verifies that your downloaded dataset matches the expected dataset.

Step 3 — Validate files

Attempt to open every image.

Check:

- unreadable/corrupted images
- zero-sized files
- unusual dimensions
- unusual channels
- missing images

Do not silently remove anything.

Produce something like:

data_quality_report.csv

Step 4 — Duplicate analysis

Check for exact duplicate images using file/image hashes.

This is important because an identical CXR should not accidentally appear in both training and testing.

Step 5 — Artefact audit

This is particularly important for your research question.

Look for:

- letters
- L/R markers
- borders
- embedded text
- unusual black/white regions
- acquisition differences
- different image sizes
- class-specific visual patterns outside lungs

Your proposal explicitly identifies text, borders, hospital-source information and acquisition signatures as possible shortcut signals. DNN.pdf

Important

Don’t automatically remove all artefacts.

Your research is partly about whether models learn from them.

First document them.

P01 output

dataset_inventory.csv
data_quality_report.csv
duplicate_report.csv
class_distribution.png

⸻

P02 — Image-mask verification

Because the dataset already supplies masks, you don’t need to generate new primary-dataset masks.

But you must ensure they are correct.

Step 1 — Match image and mask

For every:

COVID-123.png

identify its corresponding:

COVID-123_mask.png

or whatever naming convention the downloaded dataset actually uses.

Step 2 — Verify completeness

Calculate:

Total valid CXR images: X
Images with matching masks: Y
Images without masks: Z

Ideally:

X = Y
Z = 0

Step 3 — Check dimensions

For each pair:

CXR dimensions
vs
mask dimensions

Record mismatches.

Step 4 — Check masks themselves

Verify masks aren’t:

completely black
completely white
corrupted
empty

Step 5 — Visual inspection

Randomly sample images from all four classes.

Overlay the mask onto the X-ray.

You want to visually confirm:

        X-ray
          +
       Lung mask
          ↓
    Correctly aligned

Both lungs should generally be covered while background regions should mostly be excluded.

P02 output

A verified mapping:

image_path ↔ mask_path

and ideally a small mask-quality report.

⸻

P03 — Master dataset manifest

Now combine P01 and P02 into one authoritative dataset table.

Create:

dataset_manifest_v1.csv

Recommended columns:

image_id
image_path
mask_path
label
class_id
dataset

For example:

image_id image_path mask_path label class_id
COVID-1 … … COVID 0
Normal-1 … … Normal 1
LO-1 … … Lung Opacity 2
VP-1 … … Viral Pneumonia 3

Choose the class mapping once.

For example:

0 = COVID
1 = Normal
2 = Lung Opacity
3 = Viral Pneumonia

Then don’t change it between models.

This CSV becomes the foundation of your data pipeline.

⸻

P04 — Fixed train/validation/test split

This is one of the most important processes.

Your proposal explicitly requires:

70% Training
15% Validation
15% Test

using stratified sampling and a fixed random seed. CS3631_Project_Proposal_Group04.pdf

Use your T02 standard:

seed = 42

Step 1

Split:

100%
│
├──────────── 70% TRAIN
│
└──────────── 30% TEMP

Stratify using class label.

Step 2

Split TEMP:

30%
│
├──────────── 15% VALIDATION
│
└──────────── 15% TEST

Again stratify by label.

Step 3 — Verify proportions

Produce:

Class Train Validation Test
COVID
Normal
Lung Opacity
Viral Pneumonia

The proportions should remain similar.

Step 4 — Verify leakage

Check that image IDs don’t overlap.

Train ∩ Validation = empty
Train ∩ Test = empty
Validation ∩ Test = empty

If usable patient identifiers are available, also perform patient-level overlap checks. Your proposal itself does not define patient-level splitting, so don’t fabricate patient IDs if the dataset doesn’t supply usable identifiers.

Step 5 — Freeze split

Create:

split_v1.csv

containing:

image_id
image_path
mask_path
label
class_id
split

Then:

Never regenerate this split separately for ResNet50, DenseNet121, EfficientNet-B0 or ViT.

The proposal requires identical splits for fair architecture comparison. DNN.pdf

⸻

P05 — Preprocessing and augmentation

The proposal defines preprocessing as:

resize to 224×224, convert to three channels, normalize using ImageNet statistics. DNN.pdf

You actually need two transformation pipelines.

Training transform

Original CXR
↓
3-channel conversion
↓
Random resized crop / resize to 224×224
↓
Random rotation
↓
Random horizontal flip
↓
Brightness/contrast adjustment
↓
Tensor conversion
↓
ImageNet normalization

These augmentations come directly from the proposal. CS3631_Project_Proposal_Group04.pdf

Validation/test transform

No random augmentation.

Original CXR
↓
3-channel conversion
↓
Resize 224×224
↓
Tensor
↓
ImageNet normalization

Masks need special handling

If an image and mask undergo a geometric transformation together, they must remain spatially aligned.

For example:

Image rotates 10° +
Mask rotates 10°

Not:

Image rotates 10°
Mask unchanged ❌

And mask resizing should generally use nearest-neighbor-style interpolation so you don’t turn a binary mask into meaningless interpolated class values.

This becomes especially important when you start Candidate A/B/C.

⸻

P06 — PyTorch Dataset and DataLoader

Create a reusable dataset rather than putting image-loading code inside every architecture notebook.

Conceptually:

dataset = CXRDataset(
manifest="split_v1.csv",
split="train",
transform=train_transform
)

The dataset should return at minimum:

image
label

And for mask-aware experiments:

image
mask
label

Create:

train_loader
val_loader
test_loader

Train

shuffle = True
augmentation = True

Validation

shuffle = False
augmentation = False

Test

shuffle = False
augmentation = False

Connect Claude’s existing T02 utilities:

set_seed()
seed_worker()
create_generator()

so DataLoader randomness is reproducible.

⸻

P07 — Class imbalance handling

The proposal explicitly specifies class-weighted cross-entropy. DNN.pdf

Calculate weights using:

TRAINING SET ONLY

For example:

Normal
10,000 images
Viral Pneumonia
1,300 images

Without weighting, the model could prefer the majority class.

Class weighting increases the relative penalty for mistakes on underrepresented classes.

Then use:

CrossEntropyLoss(
weight=class_weights
)

Record the weights in W&B/configuration.

⸻

P08 — Stage 1 vanilla models

Now the actual deep-learning experiments start.

Your four models are:

DenseNet121 ← primary project baseline
ResNet50
EfficientNet-B0
ViT-Base

CS3631_Project_Proposal_Group04.pdf

All should use ImageNet-pretrained initialization.

Replace the final classifier with:

4 outputs

corresponding to your four disease categories.

Very important experimental rule

At this stage:

DO NOT use lung masks to suppress the image.

The vanilla models should see the standard CXR.

Why?

Because you need:

Vanilla model
VS
Shortcut-suppressed model

later.

If you alter the vanilla model using lung masking already, you lose the clean baseline comparison.

⸻

P09 — Two-phase transfer learning and HP tuning

Your proposal specifies two-phase transfer learning. DNN.pdf

Phase 1

ImageNet pretrained model
↓
Freeze backbone
↓
Train new 4-class classifier

Evaluate validation performance.

Save best checkpoint.

Phase 2

Best Phase-1 checkpoint
↓
Unfreeze final blocks
↓
Use smaller learning rate
↓
Fine-tune

Again monitor validation loss.

Your proposal specifies early stopping based on validation loss.

Hyperparameter search

The shorter proposal provides:

Learning rate:
1e-3
3e-4
1e-4
3e-5
Batch size:
16
32
64
Optimizer:
Adam
AdamW
SGD + momentum
Weight decay:
0
1e-4
1e-3
Scheduler:
Cosine
Step decay
Reduce-on-plateau

DNN.pdf

This is where W&B becomes particularly useful.

⸻

P10 — Internal baseline evaluation

After selecting the best configuration using validation data, evaluate the selected baseline on the internal test set.

Your proposal requires:

Accuracy
Precision
Recall
F1
ROC-AUC
Per-class metrics
Macro metrics
Confusion matrix

DNN.pdf

For example:

             Accuracy  Macro-F1  Macro-AUC

DenseNet121 ...
ResNet50 ...
EfficientNet ...
ViT ...

These are your Stage 1 vanilla results.

⸻

P11 — Shortcut-Suppression Module development

Now you start the novel research contribution.

The proposal defines three candidates. CS3631_Project_Proposal_Group04.pdf

Candidate A — Lung-Region Attention Module

Provided mask:

CXR ──────────────→ Backbone features
↑
Lung mask ─→ attention module
↓
Reweighted features
↓
Prediction

The mask encourages the model to emphasize lung regions and suppress background regions.

This is particularly relevant to your M1 responsibility.

⸻

Candidate B — Auxiliary Segmentation Head

                   Backbone
                      │
               Shared features
                 /          \
                ↓            ↓
       Classification    Segmentation
            head             head
             ↓                ↓
         4 classes       predicted mask

The provided dataset mask becomes the ground-truth segmentation target.

Training combines:

classification loss +
segmentation loss

⸻

Candidate C — Shortcut-Suppression Loss

Model
↓
Prediction
↓
Grad-CAM

- Provided lung mask
  ↓
  Measure activation outside lungs
  ↓
  Penalty

The objective becomes conceptually:

# Total Loss

Classification Loss +
λ × Shortcut Penalty

The proposal correctly identifies this as the higher-risk exploratory candidate. CS3631_Project_Proposal_Group04.pdf

⸻

P12 — Ablation and winner selection

Develop the candidates primarily using DenseNet121 as specified in your proposal.

Compare:

DenseNet121 Vanilla
VS
DenseNet121 + Candidate A
VS
DenseNet121 + Candidate B
VS
DenseNet121 + Candidate C

Do not choose the winner simply because it has the highest accuracy.

Your proposal’s goal is improvement in:

faithfulness

- robustness

without significant classification-performance loss. CS3631_Project_Proposal_Group04.pdf

Then select:

WINNING MODULE

⸻

P13 — Cross-architecture generalization

Now test whether your claimed model-agnostic module really generalizes.

Suppose Candidate A wins.

Run:

ResNet50
VS
ResNet50 + A
DenseNet121
VS
DenseNet121 + A
EfficientNet-B0
VS
EfficientNet-B0 + A
ViT-Base
VS
ViT-Base + A

Everything else should remain controlled.

Same:

split
seed
preprocessing
evaluation protocol

This is Stage 3 in the proposal. CS3631_Project_Proposal_Group04.pdf

⸻

P14 — Five-axis trustworthiness benchmark

Now evaluate the models using the five dimensions defined in your proposal.

Axis 1 — Classification

Accuracy
Precision
Recall
F1
ROC-AUC
Per-class results
Macro results
Confusion matrix

Axis 2 — Calibration

ECE
Brier score
Reliability diagram
Before temperature scaling
VS
After temperature scaling

Axis 3 — Explanation faithfulness

For CNNs:

Grad-CAM

For ViT:

Attention rollout

Compare explanation regions against the supplied lung mask.

Calculate the proposed quantitative lung-localization score.

Axis 4 — Efficiency

Measure:

Parameter count
FLOPs
Model size
CPU inference latency
GPU inference latency

Axis 5 — Robustness

Compare:

Internal test performance
VS
External RSNA performance

These five axes come directly from your proposal. DNN.pdf

⸻

P15 — External RSNA pipeline

Keep RSNA completely separate.

RSNA Dataset
↓
Dataset audit
↓
Required label/evaluation mapping
↓
Preprocessing
224×224
3-channel
ImageNet normalization
↓
Already-trained models
↓
External evaluation

Absolutely no:

RSNA training
RSNA fine-tuning
RSNA hyperparameter selection

because the proposal reserves it exclusively for OOD evaluation. DNN.pdf

Important unresolved issue

Your primary model predicts:

COVID
Normal
Viral Pneumonia
Lung Opacity

while your proposal describes RSNA as:

Pneumonia
Normal

The proposals don’t define the exact mapping required to make those evaluation spaces comparable. Therefore this needs to be formally designed before P15 rather than silently deciding on a mapping.

⸻

P16 — Comparative analysis

At the end you’ll have two major groups:

VANILLA
ResNet50
DenseNet121
EfficientNet-B0
ViT-Base

versus

SHORTCUT SUPPRESSED
ResNet50 + module
DenseNet121 + module
EfficientNet-B0 + module
ViT-Base + module

For each compare:

classification
calibration
faithfulness
efficiency
robustness

Your central question becomes:

Did the proposed module actually reduce shortcut reliance while maintaining useful classification performance?

⸻

P17 — Final research outputs

Eventually your results should support tables such as:

Classification

Model Accuracy Macro-F1 Macro-AUC
ResNet50
ResNet50 + Module
DenseNet121
DenseNet121 + Module

Trustworthiness

Model ECE ↓ Faithfulness ↑ AUC Drop ↓ FLOPs
Vanilla

- Module

Then you’ll have the evidence needed for the claims described in your expected contributions. CS3631_Project_Proposal_Group04.pdf

⸻

Where W&B fits

Your Claude-created T02 infrastructure wraps around almost the entire experimental part:

                     W&B
                      │
     ┌────────────────┼────────────────┐
     ↓                ↓                ↓

Stage 1 Stage 2 Stage 3
Baselines Modules Generalization
│ │ │
└────────────────┼────────────────┘
↓
Experiment history

Every official run records:

model
seed
split_version
learning rate
batch size
optimizer
augmentation
epoch
train loss
validation loss
accuracy
precision
recall
F1
AUC
best epoch

Later add calibration, faithfulness, efficiency and robustness metrics.

⸻

What you should implement NOW

Don’t ask Claude to implement P01–P17 in one huge request. That will make it much harder to verify whether the research pipeline is correct.

Your immediate sequence should be:

P01 Dataset Audit → P02 Image/Mask Verification → P03 Master Manifest → P04 Fixed 70/15/15 Split → P05 Preprocessing → P06 Dataset/DataLoaders → P07 Class Weights.

Then perform a data-pipeline smoke test:

Load train batch
↓
Display 8 images
↓
Display labels
↓
Display corresponding masks
↓
Verify image/mask transformations align
↓
Check tensor dimensions
↓
Check class weights
↓
Check deterministic loading

Once that passes, freeze the shared data pipeline.

Only then move to P08 — Stage 1 vanilla training, where your personal first architecture is ResNet50, while DenseNet121 remains the project’s primary baseline as specified in the proposal.

That gives you a clean boundary:

T02 Experiment Infrastructure ✅ DONE
P01–P07 Shared Data Pipeline ← DO THIS NOW
P08+ Deep Learning Experiments → AFTER DATA PIPELINE

This sequence preserves the methodology in both of your proposals while taking advantage of the fact that your chosen primary dataset already provides the lung masks needed by your proposed shortcut-suppression work.
