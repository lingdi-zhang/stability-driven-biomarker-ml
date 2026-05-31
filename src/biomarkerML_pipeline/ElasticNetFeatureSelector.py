from   .inner_cv  import  run_inner_cv
from sklearn.base import clone
import pandas as pd
import  numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,average_precision_score
from   sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin


class ElasticNetFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self,best_estimator,top_n=None):

        """
        Parameters
        ----------
        best_estimator : fitted sklearn pipeline or model of ElasticNet
            Typically from hyperparameter tuning (e.g., GridSearchCV.best_estimator_) or only ElasticNet without tunning
        top_n : 
            Number of top features to select based on absolute coefficients
            if None, keep all features with non zero coefficients
            default as None
        """

        self.best_estimator=best_estimator
        self.top_n = top_n
        self.model = None
        self.selected_features_ = None
        self.ranking_=None
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Fit the estimator and select top features.
        """
        self.model = clone(self.best_estimator)
        self.model.fit(X, y)
        if hasattr(self.model, "named_steps"):
            coef = np.abs(self.model.named_steps["model"].coef_).ravel()
        else:
            coef = np.abs(self.model.coef_).ravel()

        ranking = pd.DataFrame({
            "feature": X.columns,
            "importance": coef}).sort_values("importance", ascending=False)

        ranking = ranking[ranking["importance"] > 0]
        self.ranking_ = ranking
        if self.top_n is None:
            self.selected_features_ = ranking["feature"].tolist()
        elif self.top_n is not None:
            self.selected_features_ = ranking.head(self.top_n)["feature"].tolist()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features_]

    def fit_transform(self,X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)


