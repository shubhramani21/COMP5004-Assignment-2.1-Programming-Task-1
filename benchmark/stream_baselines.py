"""Loop-based reference implementations for the streaming benchmarks.

Each function computes the same result as a vectorised component of
``numcompute_stream`` but processes one sample at a time in Python. They
exist only to measure the speed-up of vectorisation and to check that
both versions agree.
"""

import numpy as np


def loop_tree_predict(tree, X):
    """Predict by walking the tree separately for every sample.

    Equivalent to ``DecisionTreeClassifier.predict``, which instead routes
    whole groups of samples with one boolean mask per node.
    """
    X = np.asarray(X, dtype=np.float64)
    predictions = []
    for row in X:
        node = tree.root_
        while not node.is_leaf:
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        predictions.append(tree.classes_[int(np.argmax(node.value))])
    return np.asarray(predictions)


def loop_welford(chunks):
    """Running per-column mean and population std, one value at a time.

    Classic single-sample Welford update; equivalent to
    ``StandardScaler.partial_fit`` applied to every chunk (without NaN).
    """
    count = None
    for chunk in chunks:
        chunk = np.asarray(chunk, dtype=np.float64)
        if count is None:
            n_features = chunk.shape[1]
            count = 0
            mean = [0.0] * n_features
            m2 = [0.0] * n_features
        for row in chunk:
            count += 1
            for j in range(n_features):
                delta = row[j] - mean[j]
                mean[j] += delta / count
                m2[j] += delta * (row[j] - mean[j])
    std = [np.sqrt(value / count) for value in m2]
    return np.asarray(mean), np.asarray(std)


def loop_confusion_matrix(chunks, n_classes):
    """Accumulate a confusion matrix one (true, predicted) pair at a time.

    Equivalent to ``StreamingConfusionMatrix`` with labels 0..n_classes-1,
    which counts each chunk with a single ``np.bincount`` call.
    """
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for y_true, y_pred in chunks:
        for actual, predicted in zip(y_true, y_pred):
            matrix[int(actual), int(predicted)] += 1
    return matrix
