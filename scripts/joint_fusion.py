"""Feature-concatenation comparison fitted on the same outer and inner folds."""
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import nested_evaluation as nested

PATIENCE = 40


def inputs(x, training, evaluation):
    spaces = nested.transform(x, training, evaluation)
    return tuple([part[:, :, None].astype(np.float32) for part in pair]
                 for pair in zip(spaces["raw"], spaces["pca"]))


def fit(tf, factory, train_inputs, val_inputs, train_y, val_y, seed, classes, epochs=None):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = factory("linear", ((697, 1), (100, 1)), len(classes))
    assert model.layers[-2].output_shape[-1] == 448
    model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss="sparse_categorical_crossentropy")
    weights = compute_class_weight("balanced", classes=classes, y=train_y)
    kwargs = {"batch_size": 32, "verbose": 0, "class_weight": dict(zip(classes, weights))}
    if epochs is None:
        stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
        history = model.fit(train_inputs, train_y, validation_data=(val_inputs, val_y),
                            epochs=nested.MAX_EPOCHS, callbacks=[stop], **kwargs)
        assert stop.best_weights is not None
        return np.empty((0, len(classes))), {"epoch": int(np.argmin(history.history["val_loss"])) + 1}
    model.fit(train_inputs, train_y, epochs=epochs, **kwargs)
    probabilities = model.predict(val_inputs, batch_size=32, verbose=0)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-5, rtol=0)
    return probabilities, {"epoch": epochs, "parameters": model.count_params()}


def run_fold(tf, factory, x, y, ids, splits, folder, classes):
    inner, dev, test = splits
    inner_inputs = [inputs(x, tr, val) for tr, val in inner]
    final_inputs = inputs(x, dev, test)
    folder.mkdir(parents=True, exist_ok=True)
    outputs = []
    for seed in nested.SEEDS:
        selected = []
        for number, ((tr, val), (fit_x, val_x)) in enumerate(zip(inner, inner_inputs), 1):
            _, details = nested.cached(folder / f"inner{number}_seed{seed}.npz", ids[val],
                                       lambda: fit(tf, factory, fit_x, val_x, y[tr], y[val], seed, classes))
            selected.append(details["epoch"])
            print(folder.parent.name, folder.name, "inner", number, seed, details, flush=True)
        epochs = int(np.median(selected))
        train_x, test_x = final_inputs
        probabilities, details = nested.cached(
            folder / f"refit_seed{seed}.npz", ids[test],
            lambda: fit(tf, factory, train_x, test_x, y[dev], None, seed, classes, epochs),
        )
        assert details["epoch"] == epochs
        outputs.append(probabilities.argmax(axis=1))
        print(folder.parent.name, folder.name, "refit", seed, details, flush=True)
    return test, np.column_stack(outputs)
