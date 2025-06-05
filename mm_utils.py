import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list

# Parameter grids for machine learning models
param_grids = {
    "svm": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf"],
        "gamma": ["auto", "scale"],
    },
    "logreg": [
        {
            "penalty": ["l1", "l2"],
            "C": [0.1, 1, 10, 100],
            "fit_intercept": [True, False],
            "solver": ["liblinear", "saga"],
        },
        {
            "penalty": ["elasticnet"],
            "C": [0.1, 1, 10, 100],
            "fit_intercept": [True, False],
            "l1_ratio": [0, 0.25, 0.5, 0.75, 1],
            "solver": ["saga"],
        },
        {
            "penalty": [None],
            "fit_intercept": [True, False],
            "solver": ["newton-cg", "lbfgs", "sag", "saga"],
        },
    ],
    "rf": {
        "n_estimators": [10, 50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
}

def load_data(filepath):
    """Load CSV data and return a DataFrame or None if missing."""
    try:
        df_ori = pd.read_csv(filepath, index_col=0)
        return df_ori
    except FileNotFoundError:
        print(f"The file {filepath} was not found.")
        return None
    except pd.errors.ParserError as e:
        print(f"Error parsing the file {filepath}: {e}")
        return None

def select_columns(df, column_names):
    """Return a DataFrame with only the selected columns."""
    return df[column_names]

def cluster_data(matrix, method="complete", metric="euclidean"):
    """Cluster rows of a matrix and return the ordering of indices."""
    cluster_rows = linkage(matrix, method=method, metric=metric)
    return leaves_list(cluster_rows)

def get_param_grid(model_name):
    """Retrieve the parameter grid for a model by name."""
    return param_grids.get(model_name.lower(), {})
