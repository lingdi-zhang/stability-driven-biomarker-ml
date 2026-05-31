from joblib import Parallel, delayed
import json
from sklearn.model_selection import StratifiedKFold
from  .config  import   format_model_parameters
from  .evaluation import  compute_metrics, create_metric_table, analyze_features_acorss_models
from  .outer_cv_analysis  import  summarize_outer_cv_results
from functools import reduce
from upsetplot import from_contents, plot,UpSet
import matplotlib.pyplot as plt
from  .outer_cv  import  run_outer_cv
import pandas as pd
from   .inner_cv  import run_inner_cv 
from sklearn.base import clone
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score
from pathlib import Path
from collections import Counter

class  BiomarkerMLPipeline():
    def __init__(self,outer_model_params,feature_selection_model_params=None,prefilter_features=False,select_top_n_features=None,output_dir=None, n_jobs_gridsearch=1, n_jobs_outer_cv=1, n_jobs_models=1):
        self.outer_model_params=outer_model_params
        self.feature_selection_model_params=feature_selection_model_params
        self.prefilter_features=prefilter_features
        self.select_top_n=select_top_n_features
        self.output_dir=output_dir
        self.gridsearch_n_jobs=n_jobs_gridsearch
        self.parallel_outer_cv_n_jobs=n_jobs_outer_cv
        self.parallel_models_n_jobs=n_jobs_models
    
    def _construct_dict(self):
        self.outer_cv_params=format_model_parameters(self.outer_model_params)
        self.models=list(self.outer_cv_params.keys())
        if self.feature_selection_model_params is not None:
            self.feature_selection_params=format_model_parameters(self.feature_selection_model_params)
            self.feature_selection_params['top_n']=self.select_top_n
        elif self.feature_selection_model_params is  None:
            self.feature_selection_params={}
        
        return  self
    
    def _biomarker_selection_across_folds(self,X,y,model_name,feature_importance_selection,cross_folds_selection,**params):
        output_dir=self.output_dir
        parallel_outer_cv_n_jobs=self.parallel_outer_cv_n_jobs
        gridsearch_n_jobs=self.gridsearch_n_jobs
        prefilter_features=self.prefilter_features
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds=["fold_1","fold_2","fold_3","fold_4","fold_5"]

        backend = "loky" if parallel_outer_cv_n_jobs!= 1 else "threading"
        with Parallel(n_jobs=parallel_outer_cv_n_jobs, backend=backend) as parallel:
            output=parallel(delayed(run_outer_cv) (X,y,train_split,test_split,prefilter_features,gridsearch_n_jobs,**params)
            for fold, (train_split,test_split) in enumerate(cv.split(X,y)))
        results=dict(zip(folds,output))        
        genes,mean_auc,sd_auc,mean_pr,sd_pr,mean_f1,sd_f1=summarize_outer_cv_results(model_name,results,feature_importance_selection,cross_folds_selection,output_dir)

        return genes,[mean_auc,sd_auc,mean_pr,sd_pr,mean_f1,sd_f1]


    def stable_biomarker_selection(self,X,y,feature_importance_selection=0.75,cross_folds_selection=0.6):
        output_dir=self.output_dir
        parallel_models_n_jobs=self.parallel_models_n_jobs
        self._construct_dict()
        model_names=[]
        tasks=[]
        
        for model_name, outer_cv_params in self.outer_cv_params.items():
            params = {"feature_selection_params": self.feature_selection_params,
                        "outer_cv_params": outer_cv_params}
            model_names.append(model_name)
            tasks.append(delayed(self._biomarker_selection_across_folds)(X, y, model_name,feature_importance_selection,cross_folds_selection,**params))
        
        backend = "loky" if parallel_models_n_jobs != 1 else "threading"
        output = Parallel(n_jobs=parallel_models_n_jobs, backend=backend)(tasks)                
        metrics_results=dict(zip(model_names,[metric for _,metric in output]))
        summary=create_metric_table(metrics_results)
        gene_list=dict(zip(model_names,[genes for genes,_ in output]))
        gene_df=analyze_features_acorss_models(gene_list,output_dir) 
        
        summary.to_csv(output_dir / "CV_model_performance.csv",index=False)
        gene_df.to_csv(output_dir / "Model_biomarker_list.csv",index=False)
        self.gene_list=gene_list

        return self


    def _prediction_with_stable_biomarkers(self,X_discovery,y_discovery,X_holdout,y_holdout,appearance_threshold=0.5,**outer_cv_params):
        gridsearch_n_jobs=self.gridsearch_n_jobs
        gene_list=self.gene_list
        all_features = []
        model_keys=self.models
        for model in model_keys:
            all_features.extend(gene_list[model])
        feature_counts = Counter(all_features)
        # convert to dataframe
        summary = pd.DataFrame({
            "gene": feature_counts.keys(),
            "appearance_count": feature_counts.values()})

        summary["percent_appearance"] = (
            summary["appearance_count"] / len(model_keys))

        # filter by threshold
        selected = summary[summary["percent_appearance"] >= appearance_threshold]

        # sort
        selected = selected.sort_values(["percent_appearance", "appearance_count", "gene"],ascending=[False, False, True])

        # keep other lists
        result = {
            "consensus_features_for_appearance_threshold": selected["gene"].tolist(),
            "intersection": gene_list["intersection"],
            "union": gene_list["union"]}

        all_metrics = []
        for gene_group,genes in result.items():
            X_discovery_selected_genes=X_discovery[genes]
            X_holdout_selected_genes=X_holdout[genes]
            best_estimator,_,_=run_inner_cv(X_discovery_selected_genes,y_discovery,gridsearch_n_jobs,**outer_cv_params)
            
            final_model=clone(best_estimator)
            final_model.fit(X_discovery_selected_genes,y_discovery)
            predict_prob=final_model.predict_proba(X_holdout_selected_genes)[:, 1]
            metrics_results=compute_metrics(y_holdout, predict_prob)
            all_metrics.append(metrics_results)
            

        metrics_df = pd.DataFrame(all_metrics)
        metrics_df['selected_gene_model']=list(result.keys())
        return metrics_df


    def prediction_with_stable_biomarkers_across_models(self,X_discovery, y_discovery, X_holdout,y_holdout,appearance_threshold=0.5):
        parallel_models_n_jobs=self.parallel_models_n_jobs
        output_dir=self.output_dir
        model_names=[]
        tasks=[]
        for model_name, outer_cv_params in self.outer_cv_params.items():
            model_names.append(model_name)
            tasks.append(delayed(self._prediction_with_stable_biomarkers)(X_discovery,y_discovery,X_holdout,y_holdout,appearance_threshold,**outer_cv_params))
        backend = "loky" if parallel_models_n_jobs != 1 else "threading"
        output = Parallel(n_jobs=parallel_models_n_jobs, backend=backend)(tasks)
        output_dict=dict(zip(model_names,output))
        summary = pd.concat([v.assign(final_prediction_model=k) for k, v in output_dict.items()],ignore_index=True)
        summary.to_csv(output_dir / "holdout_test_metrics.csv",index=False)
        
    

    
