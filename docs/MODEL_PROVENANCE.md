# Model Provenance

This file records the likely source and role of each important model artifact from the original workspace.

Path aliases:

- `ARTIFACTS/`: large model/preprocessing files downloaded separately from the cloud drive.
- `ARCHIVE/`: historical experiment outputs from the original local workspace.

## Final Model Line

| Artifact | Role | Likely source script | Notes |
| --- | --- | --- | --- |
| `ARTIFACTS/Ensemble_Final.h5` | Final stitched weighted ensemble | `scripts/build_ensemble_model.py` | Internally named `Ensemble_Final`; has inputs `Input_Raw_Spectra` and `Input_PCA_Spectra`; combines branch outputs through `Weight_A`, `Weight_B`, and `Weighted_Sum`. |
| `ARTIFACTS/Ablation_msc_cbam.h5` | Raw-spectrum/RMC branch used by ensemble | `scripts/train_raw_branch_ablation.py` | Internally named `Model_msc_cbam`; this is the Model A loaded by ensemble, weight-analysis, and robustness scripts. |
| `ARTIFACTS/Model_B_msc.h5` | PCA/PM branch used by ensemble | `scripts/train_pca_branch_ablation.py` | Internally named `Model_B_msc`; this is the Model B loaded by ensemble, weight-analysis, and robustness scripts. |
| `ARTIFACTS/scaler_A_trimmed.pkl` | Raw branch scaler | `scripts/train_raw_branch_ablation.py` | Same SHA256 as the other scaler copies in the original final artifact folder. |
| `ARTIFACTS/scaler_B_trimmed.pkl` | PCA branch scaler | `scripts/train_pca_branch_ablation.py` | Same SHA256 as the other scaler copies in the original final artifact folder. |
| `ARTIFACTS/pca_B_trimmed.pkl` | PCA transform for Model B | `scripts/train_pca_branch_ablation.py` | Required before feeding `Model_B_msc.h5`. |

The saved `Ensemble_Final.h5` was likely produced with the default `A=0.5, B=0.5` weight because `meta_model_weighted_A.pkl` is not present in the workspace. Later weight stability analysis found the best test-set weight at `A=0.51` or `A=0.52`, with accuracy `0.9271523178807947`; the 0.50 weight gives `0.9205298013245033`.

## Important Non-Final or Secondary Artifacts

| Artifact | Role | Likely source script | Notes |
| --- | --- | --- | --- |
| `ARTIFACTS/model_MSC_CBAM.h5` | Later single raw-spectrum MSC+CBAM model | Original `train_model_A copy 2.py` | Timestamp is later than the ensemble branch files, but later evaluation/robustness scripts do not use this file. |
| `ARTIFACTS/scaler_MSC_CBAM.pkl` | Scaler for `model_MSC_CBAM.h5` | Original `train_model_A copy 2.py` | Kept for the single-model branch. |
| `ARCHIVE/model_MSC_CBAM_best_fold.h5` | Best single fold from 5-fold Model A experiment | `scripts/cross_validate_raw_branch.py` | Validation experiment artifact, not the stitched final ensemble. |
| `ARCHIVE/Best_Model_B_msc.h5` and related `Best_Model_B_*` files | Best folds from 5-fold Model B experiments | `scripts/cross_validate_pca_branch.py` | Validation/ablation artifacts. |
| `ARCHIVE/Fusion_5Fold_Result.png` | 5-fold fusion comparison figure | `scripts/cross_validate_fusion.py` | Cross-validation result figure. |

## Timeline Summary

- 2025-11-18: Raw/RMC ablation models, PCA/PM ablation models, and `Ensemble_Final.h5` were created.
- 2025-11-19: `model_MSC_CBAM.h5` single-model result was created.
- 2025-11-26: 5-fold validation scripts and best-fold artifacts were created.
- 2026-01-28: Ensemble weight stability and noise robustness analyses were run.
- 2026-04-07: Additional ablation/model reading work appears in `scripts/inspect_ablation_models.py` and historical archive outputs.

## Recommended GitHub Narrative

Use the following wording in a paper/code release:

> The final evaluated ensemble combines a raw-spectrum MSC-CBAM branch (`Ablation_msc_cbam.h5`) and a PCA-spectrum MSC branch (`Model_B_msc.h5`). The two branch outputs are fused by weighted probability averaging. The repository also includes single-model and 5-fold validation experiments for comparison.

## Caution

The file `xiaorong_RMC.py` is the historical script that matches `Ablation_msc_cbam.h5`. Do not "clean up" the architecture inside this script before preserving a copy, because even small changes would no longer reproduce the saved model.
