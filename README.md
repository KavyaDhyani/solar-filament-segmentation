# Solar filament instance segmentation

A solution to the **Solar Filament Segmentation Challenge 2026** (IEEE BigData Cup, MAGFiLO dataset):
every filament in a full-disk GONG H-alpha image is segmented as a separate instance, scored with
Panoptic Quality (PQ) against each expert annotation separately.

| | out-of-fold PQ (707 training images) | public leaderboard |
|---|---|---|
| expert vs expert (for reference) | 0.340 | |
| U-Net 1024, area threshold | 0.404 | |
| U-Net 1024, learned keep rule | 0.432 | 0.37 |
| U-Net 1536, learned keep rule | 0.437 | |
| **ensemble 1024 + 1536, learned keep rule** | **0.439** | **0.39** |

The pipeline: limb fit and limb-darkening correction → two U-Nets (EfficientNetV2-S encoder,
1024 and 1536 px, 5 time-grouped folds, 8-view test-time augmentation) → averaged probability map
at full resolution → hysteresis thresholding and fragment linking inside the disk → a gradient-boosting
**keep rule** that submits a candidate only if its predicted PQ contribution exceeds PQ/2.

## Reproducing the submission

```bash
pip install -r requirements.txt
# competition data: data/MAGFiLO_1.0_Kaggle_2026/ (or set FILSEG_DATA_ROOT, or run on Kaggle)
jupyter nbconvert --to notebook --execute notebooks/02_pipeline.ipynb
```

`notebooks/02_pipeline.ipynb` loads the 10 fold models from `weights/`, predicts the 180 test images,
builds the instances, applies the stored keep rule (`weights/keep_rule.txt`, `weights/postprocess.json`)
and writes `submission.csv`. No training is needed. It takes about 30 minutes on one T4 GPU and also
runs on CPU (slowly).

Reproduction check (Kaggle, 1 x T4): the rebuilt file has 1,125 filaments against 1,124 in the submitted
file, 1,009 rows are byte-identical and the pixel IoU of all filament pixels is 0.995. The small differences
come from non-deterministic GPU convolution algorithms, which move a few probabilities across the uint8
quantisation steps and thresholds near the borders of a few filaments.

## Repository layout

```
src/filseg/            the package
  data.py              COCO annotations, per-annotator records, RLE helpers
  metrics.py           PQ exactly as in the organizers' self-evaluation notebook
  solar.py             limb fit (circle search on an edge map), limb-darkening correction
  folds.py             3-day time clusters -> grouped folds
  prep.py, augment.py  network inputs (intensity, contrast, r/R), soft consensus target, augmentation
  model.py, train.py   U-Net, mixed-precision training, out-of-fold / test prediction
  instances.py         candidate filaments, features, per-annotator IoU tables, exact selection PQ
  keep_rule.py         LightGBM keep rule and threshold sweep
  pipeline.py          inference from stored weights to submission.csv
notebooks/
  01_eda.ipynb         training-set analysis behind the design choices
  02_pipeline.ipynb    end-to-end pipeline (reproduces the submission)
weights/               fold weights of both networks, keep-rule model, post-processing settings
results/               training logs, out-of-fold PQ tables, fold assignment
dev/                   builders for the notebooks and the Kaggle kernels used for training
report/                LaTeX source and figures of the technical report (organizers' ACM template)
```

## Method

### What the data says (`notebooks/01_eda.ipynb`, training set only)

- 707 images, 1,154 expert records (1-3 experts per image), 8,199 filaments, six GONG sites.
- **Experts disagree a lot**: one expert scored against another reaches PQ 0.34. The metric scores
  every record separately, so a prediction is rewarded for matching what most experts would draw.
  Training on the *soft consensus* (share of experts covering a pixel) with generous outlines beats
  training on any single expert or on the intersection.
- **Outlines decide the score**: filaments are thin (median width 11 px at 2048 px), and eroding or
  dilating the ground truth by 1 px already costs ~20% PQ. Resizing to 1024 costs ~2%, to 512 ~8%,
  hence 1024 and 1536 px inputs and post-processing at the full 2048 px.
- **Instance separation is not the problem**: connected components of the union of the true masks
  reproduce the instances almost perfectly (PQ 0.997), so a semantic network plus connected
  components suffices.
- **Experts do not annotate at the limb**: 99% of filament centroids lie within 0.9 R.
- **Neighbouring frames leak**: images of the same day share filaments, so validation folds are
  built from 3-day time clusters (329 clusters, 5 folds).
- **Disk geometry is unreliable**: hazy frames, a bright rim, a halo and the field-stop edge break
  thresholding; an exhaustive circle search on a Sobel edge map finds the limb on every image.

### Network

- Inputs (3 channels): normalised intensity, limb-darkening-corrected contrast I / I_quiet(r), and r/R.
- U-Net (`segmentation_models_pytorch`) with an ImageNet EfficientNetV2-S encoder; BCE + soft Dice loss
  against the soft consensus; AdamW, warm-up + cosine schedule, EMA weights.
- Augmentation on the GPU: dihedral flips/rotations, small affine, gain/bias, blur, noise, haze.
- 768 px random crops at batch 8 (fits a T4 in fp16, 10 GB); folds trained in parallel on 2 GPUs.
- Two runs: 1024 px / 40 epochs / seed 0 and 1536 px / 60 epochs / seed 1, same folds.
- Prediction: 8 dihedral views per image, fold models averaged (test) or the held-out fold's model (out-of-fold).

### From probabilities to filaments

1. The two runs' maps are upsampled to 2048 px and averaged.
2. Candidates are connected components of `p > 0.5` inside the disk that reach `p > 0.6` somewhere,
   have at least 50 px and a mean r/R below 0.95; components closer than 8 px are linked into one instance.
3. **Keep rule.** Every submitted instance adds 0.5 to the PQ denominator; a hit adds its IoU to the
   numerator and turns a miss into a match. A candidate is therefore worth submitting when its expected
   IoU-weighted hit rate, averaged over the image's experts, exceeds PQ/2. A LightGBM regressor predicts
   that value from 17 candidate features (area, probability statistics, r/R, contrast, elongation,
   extent, number of linked fragments, neighbour distance, image-level statistics), trained out-of-fold
   on the same folds as the networks; the keep threshold (0.22) is swept on out-of-fold PQ, close to the
   theoretical PQ/2.

The keep rule cuts false positives from 4,628 to 2,320 at the cost of 352 matches, raising
out-of-fold PQ from 0.398 (keep everything) to 0.439.

### Validation

All numbers are pooled out-of-fold PQ with the official matching rule (IoU > 0.5, every annotator record
scored separately, `filseg.metrics` matches the organizers' notebook exactly). Per site the ensemble
scores 0.431-0.451 (Bh 0.437, Ch 0.431, Lh 0.434, Mh 0.437, Th 0.446, Uh 0.451).

## Retraining

The networks were trained on Kaggle (2 x T4). The kernels are generated by the builders in `dev/`:

```bash
python dev/build_train_notebook.py --config '{"size": 1024, "epochs": 40, "seed": 0}'
python dev/build_train_notebook.py --kernel train1536 --config '{"size": 1536, "epochs": 60, "seed": 1, "eval_every": 10}'
python dev/build_post_notebook.py --source <kernel of run 1> --source <kernel of run 2>
```

or directly with `filseg.train.run(Config(...))` and the post-processing notebook. Training one run takes
about 2.7 (1024 px) to 4.5 hours (1536 px) on 2 x T4 (the 5 folds are split across the two GPUs).

## Data use

Only the competition's training annotations were used for training and validation; the test images
were used for inference only. No external labels (including the public MAGFiLO release) were used.
