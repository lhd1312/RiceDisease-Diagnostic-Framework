"""Training, fold-local transformations, fusion, and pot-level statistics."""
import csv
import hashlib
import json
from itertools import combinations, product
from time import perf_counter

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

OUTER = tuple(range(1, 6))
INNER = tuple(range(1, 4))
SEEDS = (42, 143)
ALPHA = np.arange(2, 20) / 20
NETWORKS = ("RMC", "PM", "PM_MLP")
SVM_GRID = tuple(product((1, 10, 100), ("scale", .01, .001)))
MAX_EPOCHS = 200
PATIENCE = {"RMC": 35, "PM": 40, "PM_MLP": 40}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, header, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def cached(path, samples, fit):
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["ids"], samples)
            return saved["prediction"], json.loads(saved["config"].item())
    prediction, config = fit()
    with path.with_suffix(".tmp").open("wb") as file:
        np.savez_compressed(file, ids=samples, prediction=prediction, config=json.dumps(config))
    path.with_suffix(".tmp").replace(path)
    return prediction, config


def disease_group(labels):
    return np.where(labels == 0, 0, np.where(labels <= 5, 1, 2))


def transform(x, train, evaluate):
    scaler = StandardScaler().fit(x[train])
    raw_fit, raw_eval = scaler.transform(x[train]), scaler.transform(x[evaluate])
    reducer = PCA(n_components=100, random_state=42).fit(raw_fit)
    return {"raw": (raw_fit, raw_eval), "pca": (reducer.transform(raw_fit), reducer.transform(raw_eval))}


def train_network(tf, constructor, variant, name, seed, arrays, truth_train, truth_val, classes, epochs=None):
    started = perf_counter()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    fit_x, eval_x = (data.astype(np.float32)[:, :, None] for data in arrays)
    model = constructor(variant, fit_x.shape[1:], len(classes))
    model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss="sparse_categorical_crossentropy")
    weights = compute_class_weight("balanced", classes=classes, y=truth_train)
    params = {"batch_size": 32, "verbose": 0,
              "class_weight": dict(zip(classes.tolist(), weights.tolist()))}
    if epochs is None:
        stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE[name],
                                                restore_best_weights=True)
        history = model.fit(fit_x, truth_train, validation_data=(eval_x, truth_val),
                            epochs=MAX_EPOCHS, callbacks=[stop], **params)
        assert stop.best_weights is not None
        model.set_weights(stop.best_weights)
        loss = np.asarray(history.history["val_loss"])
        epochs = int(np.argmin(loss)) + 1
        details = {"epochs_run": len(loss), "selected_epochs": epochs, "val_loss": float(loss[epochs-1])}
    else:
        assert truth_val is None
        model.fit(fit_x, truth_train, epochs=epochs, **params)
        details = {"epochs_run": epochs, "selected_epochs": epochs}
    probability = model.predict(eval_x, batch_size=32, verbose=0)
    assert np.isfinite(probability).all() and (probability >= 0).all()
    np.testing.assert_allclose(probability.sum(axis=1), 1., atol=1e-5, rtol=0)
    details.update({"seed": seed, "seconds": round(perf_counter()-started, 2)})
    return probability, details


def svm_predict(arrays, train, params):
    c, gamma = params
    return SVC(C=c, gamma=gamma, kernel="rbf", class_weight="balanced").fit(arrays[0], train).predict(arrays[1])


def macro_f1(y, pred, classes):
    return float(f1_score(y, pred, labels=classes, average="macro", zero_division=0))


def tag(name, seed):
    return name if seed == 42 else f"{name}_seed{seed}"


def one_outer(tf, factories, task, outer, rows, x, y, pots, groups, plan, classes, output):
    test_pots, inner_map = plan
    dev = np.flatnonzero([p not in test_pots for p in pots])
    test = np.flatnonzero([p in test_pots for p in pots])
    assert np.array_equal(test, np.flatnonzero([int(row["test_fold"]) == outer for row in rows]))
    assert len(np.unique(y[dev])) == len(np.unique(y[test])) == len(classes)
    ids = np.array([row["sample_id"] for row in rows])
    folder = output / task / f"fold_{outer}"
    folder.mkdir(parents=True, exist_ok=True)
    all_names = [tag(name, seed) for seed in SEEDS for name in NETWORKS]
    inner_oof = {name: np.full((len(y), len(classes)), np.nan) for name in all_names}
    svm_oof = {space: np.full((len(SVM_GRID), len(y)), -1, dtype=int) for space in ("raw", "pca")}
    per_inner = []
    for inner in INNER:
        val = np.asarray([i for i in dev if inner_map[pots[i]] == inner], dtype=int)
        train = np.asarray([i for i in dev if inner_map[pots[i]] != inner], dtype=int)
        assert all(not set(pots[a]) & set(pots[b]) for a, b in combinations((train, val, test), 2))
        assert all(np.array_equal(np.unique(y[i]), classes) for i in (train, val))
        arrays = transform(x, train, val)
        configs = {}
        for seed in SEEDS:
            for name in NETWORKS:
                key = tag(name, seed)
                constructor, variant, space = factories[name]
                path = folder / f"inner{inner}_{key}.npz"
                fit = lambda: train_network(tf, constructor, variant, name, seed, arrays[space], y[train], y[val], classes)
                predicted, details = cached(path, ids[val], fit)
                assert np.isnan(inner_oof[key][val]).all()
                inner_oof[key][val] = predicted
                configs[key] = details
                print(task, outer, "inner", inner, key, details, flush=True)
        for space in svm_oof:
            for index, args in enumerate(SVM_GRID):
                svm_oof[space][index, val] = svm_predict(arrays[space], y[train], args)
        per_inner.append({"fold": inner, "training_pots": len(set(pots[train])), "validation_pots": len(set(pots[val])),
                          "training_observations": len(train), "validation_observations": len(val), "networks": configs})
    assert all(np.isfinite(values[dev]).all() and np.isnan(values[test]).all() for values in inner_oof.values())
    assert all((values[:, dev] >= 0).all() and (values[:, test] == -1).all() for values in svm_oof.values())
    epochs = {key: int(np.median([f["networks"][key]["selected_epochs"] for f in per_inner])) for key in all_names}
    alpha, weight_grids = {}, {}
    for seed in SEEDS:
        left, right = (inner_oof[tag(name, seed)][dev] for name in ("RMC", "PM"))
        correct = np.array([np.sum(y[dev] == (a*left+(1-a)*right).argmax(axis=1)) for a in ALPHA])
        alpha[str(seed)] = float(min(ALPHA[correct == correct.max()], key=lambda a: (round(abs(a-.5), 12), a)))
        weight_grids[str(seed)] = [[float(a), int(n)] for a, n in zip(ALPHA, correct)]
    svm_choice = {}
    for space, values in svm_oof.items():
        scores = [macro_f1(y[dev], row[dev], classes) for row in values]
        best = int(np.argmax(scores))
        svm_choice[space] = {"selected": list(SVM_GRID[best]), "pooled_inner_macro_f1": scores[best],
                             "grid": [[*arg, value] for arg, value in zip(SVM_GRID, scores)]}
    locked = {"outer_fold": outer, "development_pots": len(set(pots[dev])), "test_pots": len(set(pots[test])),
              "development_observations": len(dev), "test_observations": len(test), "inner": per_inner,
              "refit_epochs": epochs, "weights": alpha, "weight_grids": weight_grids, "svm": svm_choice}
    write_json(folder / "inner_selection.json", locked)
    arrays = transform(x, dev, test)
    probabilities = {}
    refits = {}
    for seed in SEEDS:
        for name in NETWORKS:
            key = tag(name, seed)
            constructor, variant, space = factories[name]
            fit = lambda: train_network(tf, constructor, variant, name, seed, arrays[space], y[dev], None, classes, epochs[key])
            probabilities[key], refits[key] = cached(folder / f"refit_{key}.npz", ids[test], fit)
            print(task, outer, "refit", key, refits[key], flush=True)
        left, right = (probabilities[tag(name, seed)] for name in ("RMC", "PM"))
        coefficient = alpha[str(seed)]
        probabilities[tag("RMC_PM", seed)] = (left + right)/2
        probabilities[tag("weighted", seed)] = coefficient*left + (1-coefficient)*right
    probabilities["RMC_RMC"] = (probabilities["RMC"]+probabilities["RMC_seed143"])/2
    probabilities["PM_PM"] = (probabilities["PM"]+probabilities["PM_seed143"])/2
    probabilities["cross_seed_pair_1"] = (probabilities["RMC"]+probabilities["PM_seed143"])/2
    probabilities["cross_seed_pair_2"] = (probabilities["RMC_seed143"]+probabilities["PM"])/2
    predicted = {key: values.argmax(axis=1) for key, values in probabilities.items()}
    for space in svm_choice:
        predicted[f"SVM_{space}"] = svm_predict(arrays[space], y[dev], svm_choice[space]["selected"])
    locked["refits"] = refits
    metric = {key: {"accuracy": float(np.mean(values == y[test])),
                    "macro_f1": macro_f1(y[test], values, classes)} for key, values in predicted.items()}
    locked["outer_metrics"] = metric
    write_json(folder / "fold_summary.json", locked)
    return test, predicted, locked


def pot_boot(scores, pots, groups):
    unique, inverse = np.unique(pots, return_inverse=True)
    counts = np.bincount(inverse)
    strata = np.zeros(len(unique), dtype=int)
    strata[inverse] = groups
    assert np.array_equal(strata[inverse], groups)
    right = np.column_stack([np.bincount(inverse, weights=col) for col in scores.T])
    rng = np.random.default_rng(20261119)
    draws = np.concatenate([rng.choice(ix, (10000, len(ix))) for group in np.unique(strata)
                            for ix in [np.flatnonzero(strata == group)]], axis=1)
    weights = np.asarray([np.bincount(row, minlength=len(unique)) for row in draws])
    return (weights @ right)/(weights @ counts)[:, None]


def summarize(task, rows, y, classes, pots, groups, blocks, output):
    names = list(blocks[0][1])
    assert all(list(block[1]) == names for block in blocks)
    assert np.array_equal(np.sort(np.concatenate([block[0] for block in blocks])), np.arange(len(rows)))
    predicted = np.full((len(rows), len(names)), -1, dtype=int)
    for indices, fold_predictions, _ in blocks:
        predicted[indices] = np.column_stack([fold_predictions[name] for name in names])
    assert (predicted >= 0).all()
    scores = predicted == y[:,None]
    boot = pot_boot(scores, pots, groups)
    results = {}
    for index, name in enumerate(names):
        pred = predicted[:,index]
        matrix = confusion_matrix(y, pred, labels=classes)
        assert matrix.sum() == len(rows) and np.trace(matrix) == scores[:,index].sum()
        results[name] = {"correct": int(np.trace(matrix)), "accuracy": float(scores[:,index].mean()),
                         "macro_f1": macro_f1(y, pred, classes), "kappa": float(cohen_kappa_score(y, pred)),
                         "fold_sd": float(np.std([scores[fold[0],index].mean() for fold in blocks], ddof=1)),
                         "accuracy_ci95": np.quantile(boot[:,index], [.025,.975]).tolist(),
                         "confusion_matrix": matrix.tolist(), "support": matrix.sum(axis=1).tolist(),
                         "recall": (np.diag(matrix)/matrix.sum(axis=1)).tolist()}
    pairs = []
    for seed in SEEDS:
        for baseline in ("RMC", "PM", "RMC_PM_MLP", "weighted"):
            left, right = tag(baseline, seed), tag("RMC_PM", seed)
            delta = results[right]["accuracy"]-results[left]["accuracy"]
            interval = np.quantile(boot[:,names.index(right)]-boot[:,names.index(left)],[.025,.975]).tolist()
            pairs.append([left, right, delta, *interval])
    header = ("sample_id", "pot_id", "acquisition_round", "test_fold", "truth")
    write_csv(output / f"{task}_predictions.csv", (*header,*names),
              ([*[row[key] for key in header], *values.tolist()] for row, values in zip(rows, predicted)))
    write_csv(output / f"{task}_metrics.csv", ("model","correct","observations","accuracy","macro_f1","kappa","fold_sd","ci95_low","ci95_high"),
              ((name, value["correct"],len(rows),value["accuracy"],value["macro_f1"],value["kappa"],value["fold_sd"],*value["accuracy_ci95"])
               for name, value in results.items()))
    write_json(output / f"{task}_results.json", {"observations":len(rows),"pots":len(set(pots)),"models":results,
                                                 "paired_columns":["reference","comparison","delta","low","high"],"paired":pairs,
                                                 "folds":[block[2] for block in blocks]})
    print(task, "complete", {name: round(results[name]["accuracy"]*100,2) for name in ("RMC_PM","RMC_PM_seed143","RMC_PM_MLP","RMC_PM_MLP_seed143")}, flush=True)
