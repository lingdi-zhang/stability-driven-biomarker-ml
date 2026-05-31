from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from   sklearn.linear_model import LogisticRegression
#from   CPMLogTransformer  import  CPMLogTransformer

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
        If True, run GridSearchCV. If False, fit input_model directly.

    Returns
    -------
    best_estimator : sklearn estimator
    best_params : dict or None
    best_score : float or None
    """

    pipe=Pipeline([("scaler","passthrough"),("model",input_model)])
    if tune_model==True:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        param_grid=input_param_grid
        search = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=True)
        gs=search.fit(X,y)
        return gs.best_estimator_, gs.best_params_, gs.best_score_
    else:
        return pipe, None, None

