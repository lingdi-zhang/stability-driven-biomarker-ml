from  .ElasticNetFeatureSelector  import  ElasticNetFeatureSelector
from   .inner_cv  import  run_inner_cv
import pandas as pd
from typing import Optional, List, Dict, Any

def feature_prefilter_function(X_train,y_train,X_test,y_test,n_jobs=1,**feature_selection_dict):
    """
    Run feature selection directly on all features.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : array-like
    X_test : pd.DataFrame
    y_test : array-like

    feature_selection_dict: dictionary for all parameters for feature selection

    Returns
    -------
    X_train_new: pd.DataFrame
    X_test_new: pd.DataFrame
    """
    top_n=feature_selection_dict['top_n']
    feature_selection_params=feature_selection_dict['feature_selection_params']
    best_estimator,_,_=run_inner_cv(X_train,y_train,n_jobs,**feature_selection_params)
    ranker=ElasticNetFeatureSelector(best_estimator,top_n)
    X_train_new=ranker.fit_transform(X_train,y_train)
    X_test_new=ranker.transform(X_test)
    return X_train_new, X_test_new


