# T31-B: Grad-CAM heatmap panel, same images on all four backbones

Generated locally 2026-10-03 (Apple M3 Pro, MPS, fp32) from the trained seed-42 checkpoints. Inference only.

- **Images:** 12 test images, 3 per class, chosen once with `src.modules.select_heatmap_images` from the committed
  DenseNet121 A0/A2 predictions (seed 42) and then frozen in `heatmap_image_list.csv`. Every model is shown on these
  same images; none were picked per backbone. The `reason` column records why each was chosen (correctly classified,
  A0 wrong but A2 right, or lowest-ILAR "shortcut-artifact proxy", which is a proxy and not a verified artifact).
- **Columns:** input, lung mask, then Grad-CAM for the predicted class at the post-gate tap for DenseNet121 A0 / A2 / A3,
  ResNet50 A0 / A3, EfficientNet-B0 A0 / A3 and ViT-Base A0 / A3. Same colour map and the same 0-1 scale in every
  panel; the white line is the lung mask; the EIL of that single image is printed under each panel, in red with the
  wrong label where the model misclassified it.
- **Files:** `heatmap_panel_one_per_class.jpg` (one image per class), `heatmap_panel_<class>.jpg` (all three images of
  a class), `heatmap_panel_eil.csv` (prediction and EIL for every image and model).

Read with care: these are 12 images, shown for illustration. The quantitative EIL results are the 1,000-image
figures in each backbone's comparison table. CNN heatmaps come from a 7x7 feature map and ViT-Base's from 14x14, so
ViT maps look finer for that reason alone.
