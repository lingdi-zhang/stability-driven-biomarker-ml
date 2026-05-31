from   .inner_cv  import  run_inner_cv
from  .evaluation  import  compute_metrics
import shap
import pandas as pd
import  numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score
from   sklearn.linear_model import LogisticRegression
from  .selection_workflows  import feature_prefilter_function
from sklearn.base import clone
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline


# =========================
# Feature importance
# =========================
def compute_feature_importance(
    ML_model,
    X_train_transform: pd.DataFrame) -> pd.DataFrame:
    """
    Compute feature importance for a trained model.
    """
    LINEAR_MODELS = (ElasticNet,LogisticRegression,SVC)
    TREE_MODELS = (RandomForestClassifier,XGBClassifier)
    
    if isinstance(ML_model, Pipeline):
        model = ML_model.named_steps["model"]
    else:
        model = ML_model
    
    if isinstance(model,LINEAR_MODELS):
        importance=model.coef_.ravel()
    elif isinstance(model,TREE_MODELS):
        explainer=shap.TreeExplainer(model)
        shap_values=explainer(X_train_transform)

        if hasattr(shap_values, "values"):
            shap_vals = shap_values.values
        else:
            shap_vals = shap_values

        if shap_vals.ndim == 3:  # binary classification
            shap_vals = shap_vals[:, :, 1]

        importance = np.mean(np.abs(shap_vals), axis=0)
    else:
        raise ValueError("Unsupported model_type")

    return  importance

# =========================
# Main outer CV
# =========================


def run_outer_cv(X,y,train_split,test_split, prefilter_features=False,n_jobs=1,**parameters_dict):
    """
    Run outer CV with model training + feature importance.
    """
    outer_cv_params=parameters_dict['outer_cv_params']
    feature_selection_params=parameters_dict['feature_selection_params']
    
    X_train,X_test=X.iloc[train_split], X.iloc[test_split]
    y=np.array(y)
    y_train,y_test=y[train_split],y[test_split]
    # =========================
    # feature selection
    # =========================
    if prefilter_features==True:
        X_train_new,X_test_new=feature_prefilter_function(X_train,y_train,X_test,y_test,n_jobs,**feature_selection_params)
    else:
        X_train_new=X_train
        X_test_new=X_test

    # =========================
    # Train model
    # =========================
    best_estimator,_,_=run_inner_cv(X_train_new,y_train,n_jobs,**outer_cv_params)

    final_model=clone(best_estimator)
    final_model.fit(X_train_new,y_train)
    # =========================
    # Predict
    # =========================
    predict_prob=final_model.predict_proba(X_test_new)[:, 1]
    metrics_results = compute_metrics(y_test, predict_prob)
    
    # =========================
    # Feature importance
    # =========================
    ML_model=final_model.named_steps["model"]
    X_train_transform=final_model[:-1].transform(X_train_new)
    feature_model_importance=compute_feature_importance(ML_model,X_train_transform)
    # =========================
    # Save fold result
    # =========================
    output={
        'feature_model_importance':feature_model_importance,
        'gene_list':X_train_new.columns.tolist(),
        **metrics_results}
        
    return output 
