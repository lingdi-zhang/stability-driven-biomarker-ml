import json
import pandas as pd
import statistics
from scipy.stats import iqr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from   .evaluation import  analyze_feature_stability_importance_each_model
def summarize_outer_cv_results(model_name,output,feature_importance_selection,cross_folds_selection,output_dir=None):
    """
    Summarize outer CV results:
    - plot stability
    - compute metrics
    - return final gene set
    """

    # ================================
    # 1. proccess output from outer CV
    # ================================

    auc_scores=[]
    pr_scores=[]
    f1_scores=[]
    feature_importance=[]
    folds=[]
    for fold, res in output.items():
        auc_scores.append(res['roc_auc'])
        pr_scores.append(res['pr_auc'])
        f1_scores.append(res['f1'])
        coef = np.asarray(res["feature_model_importance"]).ravel()
        feature_ranking=pd.DataFrame({'fold':fold,'gene':res['gene_list'],'coef':coef})
        feature_ranking['abs_coef']=feature_ranking['coef'].abs()
        threshold = feature_ranking["abs_coef"].quantile(feature_importance_selection)
        feature_ranking=feature_ranking[feature_ranking["abs_coef"] >= threshold]
        feature_importance.append(feature_ranking)

    feature_importance=pd.concat(feature_importance, ignore_index=True)
    nfold=len(output)
    # ========================================
    # 2. Plot feature stability and importance
    # ========================================
    selected=analyze_feature_stability_importance_each_model(model_name,feature_importance,nfold,cross_folds_selection,output_dir=output_dir)
    # =========================
    # 2. Metrics
    # =========================
    mean_auc=np.mean(auc_scores)
    sd_auc=np.std(auc_scores)
    mean_pr=np.mean(pr_scores)
    sd_pr=np.std(pr_scores)
    mean_f1=np.mean(f1_scores)
    sd_f1=np.std(f1_scores)

    # =========================
    # 3. Final gene list
    # =========================
    top_genes=selected['gene'].tolist()
    return top_genes,mean_auc,sd_auc,mean_pr,sd_pr,mean_f1,sd_f1



