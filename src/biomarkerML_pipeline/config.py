"""Parse model configuration without executing Python expressions."""
import ast

import numpy as np
import pandas as pd
from mord import LogisticAT
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

CONSTRUCTORS = {
    cls.__name__: cls for cls in (
        LogisticRegression, SVC, RandomForestClassifier, XGBClassifier,
        AdaBoostClassifier, GradientBoostingClassifier, GaussianNB, LogisticAT,
        StandardScaler,
    )
}
FUNCTIONS = {"logspace": np.logspace, "arange": np.arange}


def parse_config_value(value):
    """Accept literals and explicit, allowlisted constructor/function calls."""
    try:
        node = ast.parse(value, mode="eval").body
        return _parse_node(node)
    except (SyntaxError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid configuration value {value!r}: {exc}") from exc


def _parse_node(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        pass
    if not isinstance(node, ast.Call):
        raise ValueError("Use a literal or a supported constructor/function call")
    if isinstance(node.func, ast.Name):
        name = node.func.id
    elif (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
          and node.func.value.id == "np" and node.func.attr in FUNCTIONS):
        name = node.func.attr
    else:
        raise ValueError("Unsupported function call")
    factory = {**CONSTRUCTORS, **FUNCTIONS}.get(name)
    if factory is None or any(kw.arg is None for kw in node.keywords):
        raise ValueError(f"Unsupported constructor/function: {name}")
    return factory(*[_parse_node(arg) for arg in node.args],
                   **{kw.arg: _parse_node(kw.value) for kw in node.keywords})


def format_model_parameters(parameters: pd.DataFrame) -> dict:
    """Convert the six required named columns into pipeline parameters."""
    required = ["Model_name", "Base_model", "Parameters", "Parameter_values", "Scoring", "tune_model"]
    if not isinstance(parameters, pd.DataFrame) or parameters.empty:
        raise ValueError("Model configuration must be a nonempty DataFrame")
    missing = set(required) - set(parameters.columns)
    if missing:
        raise ValueError(f"Missing configuration columns: {sorted(missing)}")
    if parameters[["Model_name", "Base_model", "Scoring", "tune_model"]].isna().any().any():
        raise ValueError("Required configuration values cannot be missing")
    result = {}
    for name, rows in parameters.groupby("Model_name", sort=False):
        for column in ("Base_model", "Scoring", "tune_model"):
            if rows[column].nunique() != 1:
                raise ValueError(f"{name}: inconsistent {column} values")
        raw_tune = str(rows["tune_model"].iloc[0]).lower()
        if raw_tune not in ("true", "false"):
            raise ValueError(f"{name}: tune_model must be TRUE or FALSE")
        estimator = parse_config_value(str(rows["Base_model"].iloc[0]))
        if not hasattr(estimator, "fit"):
            raise ValueError(f"{name}: Base_model must be an estimator")
        tune = raw_tune == "true"
        grid = {}
        if tune:
            for column in ("Parameters", "Parameter_values"):
                if rows[column].isna().any() or rows[column].astype(str).str.strip().eq("").any():
                    raise ValueError(f"{name}: {column} cannot be blank when tuning is enabled")
            if rows["Parameters"].duplicated().any():
                raise ValueError(f"{name}: duplicate parameter names")
            for row in rows.itertuples():
                candidates = []
                for expression in str(row.Parameter_values).split(";"):
                    value = parse_config_value(expression.strip())
                    if isinstance(value, np.ndarray):
                        if value.ndim != 1:
                            raise ValueError(
                                f"{name}: {row.Parameters} requires a one-dimensional array"
                            )
                        candidates.extend(value.tolist())
                    else:
                        candidates.append(value)
                if not candidates:
                    raise ValueError(f"{name}: {row.Parameters} has no candidate values")
                grid[str(row.Parameters)] = candidates
        if isinstance(estimator, SVC):
            if estimator.kernel != "linear":
                raise ValueError(f"{name}: only linear SVC is supported; set kernel='linear'")
            if not estimator.probability:
                raise ValueError(f"{name}: SVC requires probability=True")
            if tune and any(v != "linear" for v in grid.get("model__kernel", [])):
                raise ValueError(f"{name}: only 'linear' is allowed in model__kernel")
        result[name] = dict(input_model=estimator, input_param_grid=grid,
                            scoring=str(rows["Scoring"].iloc[0]), tune_model=tune)
    return result
