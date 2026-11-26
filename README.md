# Mechanism-guided dual-stream hyperspectral network for early rice disease diagnosis

This repository contains the core evaluation code for the revised manuscript. It evaluates wavelength-domain multiscale features (RMC) and PCA-domain multiscale features (PM) from paired pot-level VNIR/SWIR spectra. The primary classifier averages the class probabilities from RMC and PM with equal weights.

## Reproduce the two classification tasks

Use Python 3.9 and the versions in `requirements.txt`. The reported runs used TensorFlow 2.10.0 with an NVIDIA GPU and scikit-learn 1.6.1. A full run trains two seeds across all outer and inner folds and can take several hours on a GPU.

```sh
python -m pip install -r requirements.txt
python scripts/cross_validate_fusion.py --data DATA --output runs/revision --audit-only
python scripts/cross_validate_fusion.py --data DATA --output runs/revision --feature-concat
```

`--audit-only` reconstructs the two cohorts and exports the pot IDs assigned to the outer test and inner validation sets without training. To run one task, pass `--tasks stage` or `--tasks early`. Add `--feature-concat` to fit the jointly trained 448-feature comparator in Table S6. The default build includes the two individual streams, the equal- and validation-weighted fusion models, same-stream and cross-seed ensembles, the PM first-block MLP replacement, and the spectral/PCA SVM comparisons.

### Input files

Supply the measured `VNIR.xlsx` and `SWIR.xlsx` workbooks used for the study. Spectral sheets use one row of band centres followed by one row per spectrum. The final worksheet in each file is named `盆号对应表` and contains the same six columns in both files, in this order: class sheet name, 1-based Excel row, observation ID, pot ID, acquisition round, and zero-based class code. The metadata and spectral rows must preserve the original order. The spectral dataset is available from the corresponding author upon reasonable request after publication, as described in the manuscript.

The script pairs sensor records, selects the 697 retained wavelengths, and applies Savitzky–Golay smoothing and standard normal variate normalisation to each observation. The standardiser and PCA are fitted within each training partition. The historical outer split is reconstructed using the original workbook labels with pot grouping and seed 42. The classifier uses the stage codes after applying the revision's 20-record BLB label correction. The matched-round task uses pots with valid observations on all three common acquisition rounds 1–3. The three inner folds select training duration and comparator settings using only outer development pots.

## Outputs and reference results

| Task | Pots | Observations | Equal fusion accuracy | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Ten-class disease-stage classification | 181 | 1,005 | 92.74% | 91.31% |
| Matched-round presymptomatic three-class classification | 141 | 423 | 91.73% | 91.93% |

The table gives the mean of the pooled outer-test metrics from training seeds 42 and 143. The runner writes cohort and fold CSVs, separate outer-fold selections, pooled prediction CSVs, per-seed metrics and seed-averaged results. Confidence intervals for accuracy and paired differences use stratified resampling of pots. Select a new output directory when changing the inputs or code. Neural results can vary with hardware and numerical libraries.

The public files are the main evaluation and selected comparators. Additional manuscript analyses and the full spectral dataset follow the access arrangements in the manuscript.
