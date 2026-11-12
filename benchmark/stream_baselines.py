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
