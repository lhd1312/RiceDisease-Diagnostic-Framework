"""Reproduce the revised paper's grouped evaluation on the paired spectral workbooks."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import nested_evaluation as nested
from spectra import load_data


KEYS = ("sample_id", "pot_id", "acquisition_round", "test_fold", "truth")
TASKS = ("stage", "early")
EARLY_DAYS = (1, 2, 3)


def make_plan(x, original_labels, pots, groups):
    """Recover the locked outer folds from the original row order and labels."""
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    outer_ids = np.zeros(len(pots), dtype=int)
    plan = {}
    for outer, (development, test) in enumerate(outer_cv.split(x, original_labels, pots), 1):
        outer_ids[test] = outer
        dev_pots, first = np.unique(pots[development], return_index=True)
        dev_groups = groups[development][first]
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=4200 + outer)
        inner_map = {
            pot: inner
            for inner, (_, validation) in enumerate(inner_cv.split(dev_pots, dev_groups), 1)
            for pot in dev_pots[validation]
        }
        test_pots = set(pots[test])
        if len(inner_map) != len(dev_pots) or test_pots.intersection(inner_map):
            raise ValueError(f"Invalid pot partitions in outer fold {outer}.")
        plan[outer] = test_pots, inner_map
    if np.any(outer_ids == 0) or sum(len(test) for test, _ in plan.values()) != len(set(pots)):
        raise ValueError("The outer folds must test each pot once.")
    return outer_ids, plan


def prepare(data_dir):
    x, original, metadata, _ = load_data(data_dir)
    if x.shape != (1005, 697):
        raise ValueError(f"Expected 1,005 spectra with 697 retained wavelengths, got {x.shape}.")
    pots = np.asarray([row[3] for row in metadata])
    days = np.asarray([row[4] for row in metadata])
    groups = nested.disease_group(original)
    by_pot = defaultdict(set)
    for pot, group in zip(pots, groups):
        by_pot[pot].add(group)
    if any(len(labels) != 1 for labels in by_pot.values()):
        raise ValueError("Each pot must belong to a single experimental group.")
    outer_ids, plan = make_plan(x, original, pots, groups)

    # Preserve the frozen folds, then apply the same stage correction as the revised run.
    stage = original.copy()
    correction = (days == 5) & (original == 8)
    if correction.sum() != 20:
        raise ValueError("Expected 20 round-5 entries with the original BLB-A3 code.")
    stage[correction] = 9
    np.testing.assert_array_equal(np.bincount(stage), [262, 60, 60, 80, 80, 176, 41, 41, 41, 164])

    # The matched-round cohort uses the original common-round field, with all three rounds
    # required per pot. The display-only date correction for Fig. 8B is separate.
    early_day_sets = defaultdict(set)
    for pot, day, label in zip(pots, days, stage):
        if day in EARLY_DAYS and label not in (5, 9):
            early_day_sets[pot].add(day)
    early = np.array([
        early_day_sets[pot] == set(EARLY_DAYS) and day in EARLY_DAYS
        for pot, day in zip(pots, days)
    ])
    expected = {(day, group): count for day in EARLY_DAYS
                for group, count in enumerate((40, 60, 41))}
    if Counter(zip(days[early], groups[early])) != expected:
        raise ValueError("Unexpected group counts in the matched-round cohort.")

    # Pooled prediction order in the original runs was by fold, then workbook row order.
    order = np.argsort(outer_ids, kind="stable")
    cohorts = {}
    for task, ids, labels, size in (
        ("stage", order, stage, 10),
        ("early", order[early[order]], groups, 3),
    ):
        rows = [dict(zip(KEYS, (metadata[i][2], pots[i], int(days[i]),
                                int(outer_ids[i]), int(labels[i])))) for i in ids]
        cohorts[task] = rows, x[ids], labels[ids], pots[ids], groups[ids], np.arange(size)
    return cohorts, plan


def export_plan(output, cohorts, plan):
    pot_groups = {row["pot_id"]: int(group) for row, group in zip(cohorts["stage"][0], cohorts["stage"][4])}
    nested.write_csv(
        output / "pot_folds.csv",
        ("outer_fold", "pot_id", "disease_group", "outer_role", "inner_validation_fold"),
        ((fold, pot, pot_groups[pot], "test" if pot in test else "development", inner_map.get(pot, ""))
         for fold, (test, inner_map) in plan.items() for pot in sorted(pot_groups)),
    )
    for task, (rows, _, _, pots, _, _) in cohorts.items():
        nested.write_csv(output / f"{task}_cohort.csv", KEYS, ([row[key] for key in KEYS] for row in rows))
        print(f"{task}: {len(rows)} observations from {len(set(pots))} pots", flush=True)


def run(tf, models, task, cohort, plan, output, include_concat):
    rows, features, truth, pots, groups, classes = cohort
    blocks = []
    for fold, (test_pots, inner_map) in plan.items():
        test, predictions, details = nested.one_outer(
            tf, models, task, fold, rows, features, truth, pots, groups,
            (test_pots, inner_map), classes, output,
        )
        folder = output / task / f"fold_{fold}"
        ids = np.asarray([row["sample_id"] for row in rows])
        for seed in nested.SEEDS:
            probabilities = []
            for model in ("RMC", "PM_MLP"):
                key = nested.tag(model, seed)
                with np.load(folder / f"refit_{key}.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["ids"], ids[test])
                    probabilities.append(saved["prediction"])
            predictions[nested.tag("RMC_PM_MLP", seed)] = (
                (probabilities[0] + probabilities[1]) / 2
            ).argmax(axis=1)
        if include_concat:
            from joint_fusion import run_fold
            from models import build_model_B_variant, build_progressive_model, feature_concat

            factory = feature_concat(build_progressive_model, build_model_B_variant)
            dev = np.flatnonzero(~np.isin(pots, list(test_pots)))
            splits = []
            for inner in nested.INNER:
                validation = np.array([inner_map[pots[i]] == inner for i in dev])
                splits.append((dev[~validation], dev[validation]))
            concat_test, concat_predictions = run_fold(
                tf, factory, features, truth, ids, (splits, dev, test),
                folder / "feature_concat", classes,
            )
            np.testing.assert_array_equal(concat_test, test)
            for seed, predictions_for_seed in zip(nested.SEEDS, concat_predictions.T):
                predictions[nested.tag("feature_concat", seed)] = predictions_for_seed
        blocks.append((test, predictions, details))
    nested.summarize(task, rows, truth, classes, pots, groups, blocks, output)
    export_seed_average(output, task, rows, truth, pots, groups)


def export_seed_average(output, task, rows, truth, pots, groups):
    """Average metrics across training seeds and bootstrap paired accuracy differences by pot."""
    report_path = output / f"{task}_results.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    predictions = nested.read_csv(output / f"{task}_predictions.csv")
    folds = np.array([int(row["test_fold"]) for row in rows])
    names = [
        name for name in ("RMC", "PM", "RMC_PM", "weighted", "RMC_PM_MLP", "feature_concat")
        if nested.tag(name, nested.SEEDS[0]) in report["models"]
    ]
    columns = {}
    summary = {}
    for name in names:
        tags = [nested.tag(name, seed) for seed in nested.SEEDS]
        columns[name] = np.array([[int(row[tag]) == label for tag in tags]
                                  for row, label in zip(predictions, truth)]).mean(axis=1)
        summary[name] = {
            "accuracy": float(columns[name].mean()),
            "macro_f1": float(np.mean([report["models"][tag]["macro_f1"] for tag in tags])),
            "fold_sd": float(np.std([columns[name][folds == fold].mean()
                                     for fold in nested.OUTER], ddof=1)),
        }
    for name in ("RMC_RMC", "PM_PM"):
        columns[name] = np.array([int(row[name]) == label for row, label in zip(predictions, truth)])
    all_names = list(columns)
    boot = nested.pot_boot(np.column_stack(list(columns.values())), pots, groups)
    for name in summary:
        summary[name]["accuracy_ci95"] = np.quantile(boot[:, all_names.index(name)], [.025, .975]).tolist()
    reference = all_names.index("RMC_PM")
    comparisons = {}
    for name, outcomes in columns.items():
        if name != "RMC_PM":
            delta = boot[:, reference] - boot[:, all_names.index(name)]
            comparisons[name] = {
                "equal_minus_comparator_accuracy": float((columns["RMC_PM"] - outcomes).mean()),
                "ci95": np.quantile(delta, [.025, .975]).tolist(),
            }
    nested.write_json(output / f"{task}_seed_average.json",
                      {"models": summary, "paired_comparisons": comparisons})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory containing VNIR.xlsx and SWIR.xlsx")
    parser.add_argument("--output", type=Path, default=Path("runs/revision"))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--audit-only", action="store_true", help="Export cohort and fold assignments without fitting")
    parser.add_argument("--feature-concat", action="store_true", help="Also fit the jointly trained Table S6 comparator")
    args = parser.parse_args()
    cohorts, plan = prepare(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output / "protocol.json"
    script_root = Path(__file__).resolve().parent
    protocol = {
        "inputs": {name: nested.digest(args.data / name) for name in ("VNIR.xlsx", "SWIR.xlsx")},
        "code": {path.name: nested.digest(path) for path in sorted(script_root.glob("*.py"))},
        "seeds": list(nested.SEEDS),
        "outer_folds": len(nested.OUTER),
        "inner_folds": len(nested.INNER),
    }
    if lock.exists() and json.loads(lock.read_text(encoding="utf-8")) != protocol:
        raise SystemExit("Inputs or code changed. Choose a different --output directory.")
    if not lock.exists():
        nested.write_json(lock, protocol)
    export_plan(args.output, cohorts, plan)
    if args.audit_only:
        return

    import tensorflow as tf
    from models import build_model_B_variant, build_progressive_model, replace_msc

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    factories = {
        "RMC": (build_progressive_model, "msc_cbam", "raw"),
        "PM": (build_model_B_variant, "msc", "pca"),
        "PM_MLP": (replace_msc(build_model_B_variant), "mlp", "pca"),
    }
    for task in dict.fromkeys(args.tasks):
        run(tf, factories, task, cohorts[task], plan, args.output, args.feature_concat)


if __name__ == "__main__":
    main()
