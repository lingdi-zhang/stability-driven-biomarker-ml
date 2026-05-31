from biomarkerML_pipeline.load_demo_breat_cancer_data import load_demo_data
from biomarkerML_pipeline import BiomarkerMLPipeline
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    output_dir = PROJECT_ROOT/"outputs"/"breast_cancer_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    X_discovery, X_holdout, y_discovery, y_holdout = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42)

    outer_cv_params = pd.read_csv(
        PROJECT_ROOT / "configs" / "model_params.txt",
        sep="\t"
    )

#    feature_selection_params = pd.read_csv("configs/feature_selection_params.txt",sep="\t")

    run_pipeline=BiomarkerMLPipeline(outer_cv_params,output_dir=output_dir,n_jobs_models=3) 
    run_pipeline.stable_biomarker_selection(X_discovery,y_discovery,feature_importance_selection=0.75,cross_folds_selection=0.6)
    run_pipeline.prediction_with_stable_biomarkers_across_models(X_discovery, y_discovery, X_holdout,y_holdout,appearance_threshold=0.5)
