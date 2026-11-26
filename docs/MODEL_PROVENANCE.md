# Evaluation provenance

## Revised evaluation

- `scripts/spectra.py` loads and aligns the two measured sensor workbooks. It retains the manuscript's spectral bands, concatenates them in sensor order, and applies the 11-band, third-order Savitzky–Golay filter followed by observation-wise SNV.
- `scripts/cross_validate_fusion.py` reconstructs the frozen outer folds from the original row order and stage codes. It applies the stage-code correction after the outer folds are assigned, and uses the eligible complete-round subset for early diagnosis.
- `scripts/models.py` contains the original RMC and PM model builders extracted from the historical training scripts, plus the PM first-block MLP replacement and joint feature concatenation.
- `scripts/nested_evaluation.py` implements the inner selection, the outer refits and test predictions, the ensemble comparisons, and the pot-level bootstrap.
- `scripts/joint_fusion.py` implements the optional Table S6 comparison on the same outer and inner folds.

The main score pools one held-out prediction per observation per seed. The two seed-wise scores are then averaged. Model fitting and selection use only the corresponding development data. Equal fusion uses a fixed weight of 0.50 for each stream. The validation-weighted comparison selects its weight from inner validation predictions.

The supplementary temporal figure uses the confirmed acquisition days for the affected RB observations. Its display-time correction does not alter the original fold assignments, the corrected stage codes, or membership of the matched-round classification cohort.

## Historical implementation

The scripts from the original submission are available in the [pre-revision commit](https://github.com/lhd1312/RiceDisease-Diagnostic-Framework/tree/f2d38c72a5187b8ccb4d62237e4753ea9fffc8de). Those scripts include the original observation-level evaluation, which was superseded by the grouped protocol described in the revision.
