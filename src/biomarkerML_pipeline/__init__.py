"""
biomarkerML_pipeline

Machine learning framework for stability-driven biomarker discovery,
cross-validation stability assessment, and cross-model consensus analysis.
"""

from .biomarkerML_pipeline import BiomarkerMLPipeline
from .config import format_model_parameters
from .ElasticNetFeatureSelector import ElasticNetFeatureSelector
from .load_demo_breast_cancer_data import load_demo_data

__version__ = "0.1.0"
__author__ = "Lingdi Zhang"

__all__ = [
    "BiomarkerMLPipeline",
    "ElasticNetFeatureSelector",
    "format_model_parameters",
    "load_demo_data",
]
