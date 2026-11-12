"""Sequential pipelines and horizontal feature unions.

Components follow fit(X, y=None) and transform(X) conventions.
An optional fit_transform(X, y=None) method is used when available.
"""
import numpy as np


def _has_method(obj, name):
    return callable(getattr(obj, name, None))

def _check_unique_objects(obj, seen, *, path):
    """Reject repeated objects within a built-in composition tree.

    Traverse nested Pipeline and FeatureUnion objects. Treat custom
    transformers and estimators as leaves without inspecting their
    internal attributes.

    Record objects before descending so cycles are also rejected.
    """
    object_id = id(obj)

    if object_id in seen:
        raise ValueError(
            f"Step {path!r} reuses an object already present "
            "in this composition; use a separate object for each step."
        )

    seen.add(object_id)

    if isinstance(obj, Pipeline):
        children = obj.steps
    elif isinstance(obj, FeatureUnion):
        children = obj.transformers
    else:
        return

    for child_name, child in children:
        _check_unique_objects(child,seen,path=f"{path}.{child_name}",)

def _validate_named_steps(steps, *, name):
    if isinstance(steps, (str, bytes)) or not isinstance(steps, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple of (name, object) pairs.")

    if len(steps) == 0:
        raise ValueError(f"{name} must contain at least one step.")

    names = set()
    objects = set()

    for index, step in enumerate(steps):
        if not isinstance(step, tuple):
            raise TypeError(f"{name}[{index}] must be a (name, object) tuple.")

        if len(step) != 2:
            raise ValueError(
                f"{name}[{index}] must contain exactly a name and an object; "
                f"got {len(step)} items."
            )

        step_name, obj = step

        if not isinstance(step_name, str):
            raise TypeError(f"The name of {name}[{index}] must be a string.")

        if step_name == "":
            raise ValueError(f"The name of {name}[{index}] must not be empty.")

        if step_name in names:
            raise ValueError(f"Step names must be unique; {step_name!r} is repeated.")

        # Check both this object and objects inside nested compositions.
        _check_unique_objects(obj,objects,path=f"{name}.{step_name}",)

        names.add(step_name)

    return list(steps)


def _fit_transform_step(step, X, y):
    if _has_method(step, "fit_transform"):
        return step.fit_transform(X, y)

    step.fit(X, y)
    return step.transform(X)


class Pipeline:
    """Chain transformations and an optional final estimator.

    Parameters
    ----------
    steps : list or tuple of (str, object) tuples
        Non-empty ordered steps with unique, non-empty names.
        Intermediate steps must implement fit and transform.
        The final step must implement fit; its other methods determine
        whether transform, predict and fit_transform are supported.

    Attributes
    ----------
    steps : list
        A shallow copy of the supplied sequence.
    named_steps : dict
        Mapping from step names to the original component objects.

    Raises
    ------
    TypeError
        If step definitions have unsupported types or a component lacks
        a required callable method.
    ValueError
        If steps are empty, pairs have the wrong length, names are empty
        or repeated, or a component object is reused in the composition.

    Notes
    -----
    Components are not cloned. Fitting changes their state in place.
    Reuse checks traverse nested Pipeline and FeatureUnion objects;
    custom components are treated as leaves. These checks run only
    during construction. Do not mutate the composition afterward.

    Input validation, output shapes and input mutation depend on the
    components. The pipeline does not copy X or y before forwarding them.
    Component exceptions propagate to the caller.

    Execution time is the sum of the invoked component costs, plus
    linear overhead in the number of steps. Memory depends on component
    state, intermediate outputs and component workspace.
    """
    def __init__(self, steps):
        self.steps = _validate_named_steps(steps, name="steps")
        self.named_steps = dict(self.steps)
        self._is_fitted = False

        for step_name, obj in self.steps[:-1]:
            if not (_has_method(obj, "fit") and _has_method(obj, "transform")):
                raise TypeError(
                    f"Intermediate step {step_name!r} must provide "
                    "fit and transform methods."
                )

        last_name, last = self.steps[-1]

        if not _has_method(last, "fit"):
            raise TypeError(f"Final step {last_name!r} must provide a fit method.")

    @property
    def _final_step(self):
        return self.steps[-1][1]

    def _require(self, method):
        """Check that the final step supports method and the pipeline is fitted."""
        last_name, last = self.steps[-1]

        if not _has_method(last, method):
            raise TypeError(
                f"Pipeline does not support {method} because final step "
                f"{last_name!r} has no {method} method."
            )

        if not self._is_fitted:
            raise RuntimeError(f"Pipeline must be fitted before {method}.")

    def _fit_intermediate(self, X, y):
        # A failed fit leaves the pipeline unfitted.
        self._is_fitted = False

        for _, step in self.steps[:-1]:
            X = _fit_transform_step(step, X, y)

        return X

    def _transform_intermediate(self, X):
        for _, step in self.steps[:-1]:
            X = step.transform(X)

        return X

    def fit(self, X, y=None):
        """Fit intermediate transformations, then fit the final step.

        Parameters
        ----------
        X : array-like
            Input accepted by the first step. Each intermediate output
            becomes the input to the next step.
        y : object, optional
            Target data passed unchanged to every fitting operation.

        Returns
        -------
        Pipeline
            This pipeline instance.

        Notes
        -----
        Intermediate steps use fit_transform when available, otherwise
        fit followed by transform. The final step uses fit only.
        A component failure leaves the pipeline marked as unfitted;
        changes already made to component state are not rolled back.
        """
        Xt = self._fit_intermediate(X, y)
        self._final_step.fit(Xt, y)
        self._is_fitted = True

        return self

    def transform(self, X):
        """Transform X through every step without refitting.

        Parameters
        ----------
        X : array-like
            Input accepted by the first step.

        Returns
        -------
        object
            The final step's transform output.

        Raises
        ------
        TypeError
            If the final step has no callable transform method.
        RuntimeError
            If transformation is supported but the pipeline is unfitted.

        Notes
        -----
        Output shape, dtype and memory ownership depend on the components.
        """
        self._require("transform")
        return self._final_step.transform(self._transform_intermediate(X))

    def predict(self, X):
        """Transform X through intermediate steps and predict.

        Parameters
        ----------
        X : array-like
            Input accepted by the first step.

        Returns
        -------
        object
            The final step's predict output.

        Raises
        ------
        TypeError
            If the final step has no callable predict method.
        RuntimeError
            If prediction is supported but the pipeline is unfitted.

        Notes
        -----
        No step is refitted. Prediction shape and dtype are determined
        by the final estimator.
        """
        self._require("predict")
        return self._final_step.predict(self._transform_intermediate(X))

    def fit_transform(self, X, y=None):
        """Fit and transform X through every step.

        Parameters
        ----------
        X : array-like
            Input accepted by the first step.
        y : object, optional
            Target data passed unchanged to every fitting operation.

        Returns
        -------
        object
            The final step's transformed output.

        Raises
        ------
        TypeError
            If the final step has no callable transform method.

        Notes
        -----
        Each step uses fit_transform when available, otherwise fit
        followed by transform. The pipeline is marked fitted only after
        all operations succeed.

        The final-step capability check occurs before fitting starts.
        Once fitting starts, a failure leaves the pipeline marked as
        unfitted; component state changes are not rolled back.
        """
        last_name, last = self.steps[-1]

        if not _has_method(last, "transform"):
            raise TypeError(
                "Pipeline does not support fit_transform because final step "
                f"{last_name!r} has no transform method."
            )

        Xt = self._fit_intermediate(X, y)
        result = _fit_transform_step(last, Xt, y)
        self._is_fitted = True

        return result


class FeatureUnion:
    """Apply transformers to the same input and concatenate their outputs.

    Parameters
    ----------
    transformers : list or tuple of (str, object) tuples
        Non-empty ordered branches with unique, non-empty names.
        Each object must implement fit and transform.

    Attributes
    ----------
    transformers : list
        A shallow copy of the supplied sequence.
    named_transformers : dict
        Mapping from branch names to the original transformer objects.

    Raises
    ------
    TypeError
        If branch definitions have unsupported types or a transformer
        lacks a required callable method.
    ValueError
        If branches are empty, pairs have the wrong length, names are
        empty or repeated, or an object is reused in the composition.

    Notes
    -----
    Branches execute sequentially on the same X and y objects.
    Transformers are not cloned, and inputs are not copied. Branches
    should avoid modifying shared inputs.

    Reuse checks traverse nested Pipeline and FeatureUnion objects
    during construction. Custom components are treated as leaves.
    Do not mutate the composition afterward.

    Every transformed output must be 2D and have len(X) rows.
    Columns follow branch order. NumPy determines the common output
    dtype when concatenating the branch outputs.
    Component exceptions propagate to the caller.
    """
    def __init__(self, transformers):
        self.transformers = _validate_named_steps(transformers, name="transformers")
        self.named_transformers = dict(self.transformers)
        self._is_fitted = False

        for branch_name, obj in self.transformers:
            if not (_has_method(obj, "fit") and _has_method(obj, "transform")):
                raise TypeError(
                    f"Branch {branch_name!r} must provide fit and transform methods."
                )

    def _n_samples(self, X):
        try:
            return len(X)
        except TypeError as exc:
            raise TypeError("X must be a 2D array-like of samples.") from exc

    def _combine(self, outputs, n_samples):
        arrays = []

        for (branch_name, _), output in zip(self.transformers, outputs):
            array = np.asarray(output)

            if array.ndim != 2:
                raise ValueError(
                    f"Branch {branch_name!r} must return a 2D array; "
                    f"got {array.ndim}D."
                )

            if array.shape[0] != n_samples:
                raise ValueError(
                    f"Branch {branch_name!r} returned {array.shape[0]} rows; "
                    f"expected {n_samples}."
                )

            arrays.append(array)

        return np.hstack(arrays)

    def fit(self, X, y=None):
        """Fit every branch on the same input and return this union.

        Parameters
        ----------
        X : array-like
            Input accepted by every branch.
        y : object, optional
            Target data passed unchanged to each branch's fit method.

        Returns
        -------
        FeatureUnion
            This union instance.

        Notes
        -----
        This method does not transform or validate branch output shapes.
        A failed fit leaves the union marked as unfitted; prior branch
        state changes are not rolled back.

        Time is the sum of branch fitting costs plus O(b) dispatch
        overhead for b branches. Memory depends on the branches.
        """
        self._is_fitted = False

        for _, branch in self.transformers:
            branch.fit(X, y)

        self._is_fitted = True

        return self

    def transform(self, X):
        """Transform X with every branch and concatenate columns.

        Parameters
        ----------
        X : array-like
            Input with len(X) samples, accepted by every branch.

        Returns
        -------
        numpy.ndarray, shape (n_samples, total_features)
            Concatenated branch outputs in their configured order.
            The result uses a common NumPy dtype and newly allocated
            array storage. Object elements, if present, are not deep-copied.

        Raises
        ------
        RuntimeError
            If the union has not been fitted successfully.
        TypeError
            If X has no sample count.
        ValueError
            If a branch output is not 2D or has a different row count.

        Notes
        -----
        No branch is refitted. For b branches and S total output cells,
        time is the sum of branch transformation costs plus O(b + S).
        Additional storage is O(b + S), excluding branch workspace,
        with all branch outputs retained until concatenation.
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureUnion must be fitted before transform.")

        n_samples = self._n_samples(X)
        outputs = [branch.transform(X) for _, branch in self.transformers]

        return self._combine(outputs, n_samples)

    def fit_transform(self, X, y=None):
        """Fit every branch and concatenate the transformed outputs.

        Parameters
        ----------
        X : array-like
            Input with len(X) samples, accepted by every branch.
        y : object, optional
            Target data passed unchanged to each fitting operation.

        Returns
        -------
        numpy.ndarray, shape (n_samples, total_features)
            Outputs concatenated in branch order using a common NumPy
            dtype and newly allocated array storage. Object elements
            are not deep-copied.

        Raises
        ------
        TypeError
            If X has no sample count.
        ValueError
            If a branch output is not 2D or has a different row count.

        Notes
        -----
        Each branch uses fit_transform when available, otherwise fit
        followed by transform. The union is marked fitted only after
        fitting, transformation and concatenation all succeed.
        Failure leaves it unfitted without rolling back branch state.

        For b branches and S total output cells, time is the sum of
        branch operation costs plus O(b + S). Additional storage is
        O(b + S), excluding branch workspace and fitted state.
        """
        self._is_fitted = False

        n_samples = self._n_samples(X)
        outputs = [_fit_transform_step(branch, X, y) for _, branch in self.transformers]
        result = self._combine(outputs, n_samples)

        self._is_fitted = True

        return result

