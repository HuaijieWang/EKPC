import numpy as np


def tensor2numpy(value):
    return value.detach().cpu().numpy()


def accuracy(y_pred, y_true, nb_old, task_sizes):
    if len(y_pred) != len(y_true) or not len(y_true):
        raise ValueError(
            "Prediction and target arrays must have the same nonzero length"
        )
    boundaries = np.cumsum([0, *task_sizes])
    predicted_tasks = np.searchsorted(boundaries[1:], y_pred, side="right")
    true_tasks = np.searchsorted(boundaries[1:], y_true, side="right")
    results = {
        "total": float(np.round(100 * np.mean(y_pred == y_true), 2)),
        "task_id": float(np.round(100 * np.mean(predicted_tasks == true_tasks), 2)),
    }
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        selected = (y_true >= start) & (y_true < end)
        if selected.any():
            results[f"{start:02d}-{end - 1:02d}"] = float(
                np.round(100 * np.mean(y_pred[selected] == y_true[selected]), 2)
            )
    for name, selected in (("old", y_true < nb_old), ("new", y_true >= nb_old)):
        results[name] = (
            float(np.round(100 * np.mean(y_pred[selected] == y_true[selected]), 2))
            if selected.any()
            else 0.0
        )
    return results


def split_images_labels(images):
    paths, labels = zip(*images)
    return (np.asarray(paths), np.asarray(labels))
