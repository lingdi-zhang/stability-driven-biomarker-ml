from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from .validation import validate_cv_counts
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.base import clone

def run_inner_cv(X,y,n_jobs,input_model,input_param_grid,scoring,tune_model=True):
    """
    Tune model hyperparameters using inner cross-validation.

    Parameters
    ----------
    X : pd.DataFrame
        Training feature matrix.

    y : np.ndarray
        Training labels.

    input_model : sklearn estimator

    input_param_grid : dict, optional
        Hyperparameter grid for GridSearchCV.

    scoring : str
        Scoring metric used in GridSearchCV.

    tune_model : bool
        If True, run GridSearchCV. If False, return a pipeline using
        input_model as configured, ignoring input_param_grid.

    Returns
    -------
    best_estimator : sklearn estimator
    best_params : dict or None
    best_score : float or None
    """

    _validate_svc(input_model)

    pipe=Pipeline([("scaler", "passthrough"), ("model", input_model)])
    if tune_model==True:
        for candidate in ParameterGrid(input_param_grid):
            _validate_svc(clone(pipe).set_params(**candidate))
        validate_cv_counts(y)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        param_grid=input_param_grid
        search = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            error_score="raise")
        gs=search.fit(X,y)
        return gs.best_estimator_, gs.best_params_, gs.best_score_
    else:
        return pipe, None, None


def _validate_svc(estimator):
    """Validate SVC settings, including estimators nested in a pipeline."""
    estimators = [estimator]
    if hasattr(estimator, "get_params"):
        estimators.extend(estimator.get_params(deep=True).values())
    for model in estimators:
        if isinstance(model, SVC):
            if model.kernel != "linear":
                raise ValueError("Only linear SVC is supported. Set kernel='linear'.")
            if not model.probability:
                raise ValueError("SVC requires probability=True")
