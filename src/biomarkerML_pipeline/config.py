import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from  numpy import logspace
from numpy import arange
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from   sklearn.linear_model import LogisticRegression
from  sklearn.naive_bayes import GaussianNB
from mord import LogisticAT

def format_model_parameters(parameters:pd.DataFrame) -> dict:

    """
    Convert user input parameters into pipeline-ready parameters.
    """
    model_dict=dict()
    models=list(set(parameters[parameters.columns[0]]))
    for item in models:
        parameters_data=parameters.loc[parameters[parameters.columns[0]]==item]
        model_parameters=parameters_data[parameters_data.columns[2]]
        model_parameter_values=parameters_data[parameters_data.columns[3]].str.split(";").tolist()
        
        for entry in range(len(model_parameter_values)):
#            if parameters_data[parameters_data.columns[4]].tolist()[entry] != "function":
            model_parameter_values[entry]=[eval(item) for item in model_parameter_values[entry]]

        parameters_list=dict(zip(model_parameters,model_parameter_values))
        model=list(set(parameters_data[parameters_data.columns[1]]))[0]
        scoring=parameters_data[parameters_data.columns[4]].tolist()[0]
        tune_model=parameters_data[parameters_data.columns[5]].tolist()[0]
        value_list={'input_model':eval(model),'input_param_grid':parameters_list,'scoring':scoring,'tune_model':tune_model}
        model_dict[item]=value_list

    return model_dict


