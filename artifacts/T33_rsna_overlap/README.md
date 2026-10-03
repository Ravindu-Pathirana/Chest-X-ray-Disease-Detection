# T33: RSNA is not an external test set for these models

Run on Kaggle 2026-10-03 with `notebooks/T33_rsna_overlap_check.ipynb` (executed copy: `T33_executed_notebook.ipynb`).
Every primary image and every RSNA challenge image was reduced to a 64x64 brightness- and contrast-normalised
fingerprint; each primary image was matched to its most similar RSNA image.

## Result

- 14,863 of the 21,165 primary images are pixel-identical to an RSNA image (similarity 0.9994 or higher; the
  highest similarity of any unmatched image is 0.966). Every match is to a distinct RSNA image and carries the
  matching RSNA label.
- All 6,012 primary Lung_Opacity images are RSNA "Lung Opacity" images. 8,851 of the 10,192 primary Normal images
  are RSNA "Normal" images. COVID and Viral Pneumonia have no RSNA matches.
- Of the 26,684 RSNA images in `artifacts/segmentation/external_manifest_v1.csv`, 10,417 are in the primary training
  split, 2,225 in validation and 2,221 in test. The only RSNA images the models never saw are the 11,821 of the
  class "No Lung Opacity / Not Normal", which the manifest labels Normal although they are abnormal.

RSNA therefore cannot be reported as out-of-distribution or external validation for models trained on this dataset.

## The dataset's source metadata is wrong per file

`Normal.metadata.xlsx` records NORMAL-8852 to NORMAL-10192 as the 1,341 images that did not come from RSNA. By
pixels, the non-RSNA Normal images are Normal-63 to Normal-1403. The counts in the metadata are right; the file
names they are attached to are not. Use `primary_image_source_verified.csv`, not the metadata file, for any
analysis by image source.

## Files

- `primary_image_source_verified.csv`: one row per primary image with its split, what the metadata says, whether
  it matches an RSNA image, and the RSNA patient id and class.
- `rsna_exposure_by_class_corrected.csv`, `overlap_summary_corrected.json`: counts computed from all pixel matches.
- `kaggle_output/`: the notebook's files as downloaded. Its `rsna_exposure_by_class.csv`, the `rsna_images_*`
  fields of its `overlap_summary.json` and its printed verdict counted only images the metadata called RSNA, so
  they understate the overlap by 1,341 images. The 341 MB fingerprint file was not kept.
