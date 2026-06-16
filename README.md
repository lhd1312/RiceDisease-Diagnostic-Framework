# A fine-grained diagnostic framework for subclinical rice disease detection: integrating spatiotemporal infection dynamics and mechanism alignment

## What Is Included

```text
.
├── README.md
├── requirements.txt
├── scripts/
│   ├── train_raw_branch_ablation.py
│   ├── train_pca_branch_ablation.py
│   ├── evaluate_weighted_ensemble.py
│   ├── build_ensemble_model.py
│   ├── analyze_ensemble_weights.py
│   ├── analyze_noise_robustness.py
│   ├── analyze_noise_robustness_snr.py
│   ├── cross_validate_raw_branch.py
│   ├── cross_validate_pca_branch.py
│   ├── cross_validate_fusion.py
│   └── inspect_ablation_models.py
└── docs/
    └── MODEL_PROVENANCE.md
```

## Final Model Line

The final evaluated model line is a weighted ensemble of two branches:

- Raw-spectrum branch: `Ablation_msc_cbam.h5`
- PCA-spectrum branch: `Model_B_msc.h5`

The ensemble evaluation scripts load:

- `Ablation_msc_cbam.h5`
- `Model_B_msc.h5`
- `scaler_A_trimmed.pkl`
- `scaler_B_trimmed.pkl`
- `pca_B_trimmed.pkl`

`Ensemble_Final.h5` is a stitched Keras model built from the same two branches. Because it contains Lambda layers, the branch models plus explicit evaluation scripts are safer for reproducibility across TensorFlow/Keras versions.

## External Files

After downloading artifacts from the cloud drive, place the required files like this:

```text
.
├── Ablation_msc_cbam.h5
├── Model_B_msc.h5
├── scaler_A_trimmed.pkl
├── scaler_B_trimmed.pkl
├── pca_B_trimmed.pkl
└── DATA/
    ├── VNIR.xlsx
    └── SWIR.xlsx
```

Optional artifacts:

- `Ensemble_Final.h5`
- `Ablation_baseline.h5`
- `Ablation_msc.h5`
- `Model_B_baseline.h5`
- `Model_B_cbam.h5`
- `Model_B_msc_cbam.h5`
- `model_MSC_CBAM.h5`
- `scaler_MSC_CBAM.pkl`

## Data Format

The scripts expect two Excel files:

- `DATA/VNIR.xlsx`
- `DATA/SWIR.xlsx`

Expected sheet names:

- `CK`
- `DWB_01_Resistant`
- `DWB_02_Resistant`
- `DWB_03_Resistant`
- `DWB_04_Resistant`
- `DWB_05_Susceptible`
- `BYK_01_Resistant`
- `BYK_02_Resistant`
- `BYK_03_Resistant`
- `BYK_04_Susceptible`

## Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run commands from the repository root, not from inside `scripts/`, because the historical scripts use relative paths such as `DATA/VNIR.xlsx` and `Ablation_msc_cbam.h5`.

Train the two branch model families:

```bash
python scripts/train_raw_branch_ablation.py
python scripts/train_pca_branch_ablation.py
```

Evaluate the weighted ensemble:

```bash
python scripts/evaluate_weighted_ensemble.py
```

Build the stitched ensemble H5:

```bash
python scripts/build_ensemble_model.py
```

Additional analyses:

```bash
python scripts/analyze_ensemble_weights.py
python scripts/analyze_noise_robustness.py
python scripts/analyze_noise_robustness_snr.py
```

5-fold validation experiments:

```bash
python scripts/cross_validate_raw_branch.py
python scripts/cross_validate_pca_branch.py
python scripts/cross_validate_fusion.py
```

## Notes

- Model weights, preprocessors, and Excel data are ignored by `.gitignore`.
- `Ablation_msc_cbam.h5` is about 103 MB, which exceeds GitHub's normal 100 MB file limit.
- Use a cloud drive, GitHub Releases, Zenodo, or Git LFS for large artifacts.
- See `docs/MODEL_PROVENANCE.md` for the mapping between historical filenames and final model usage.
