import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import pandas as pd
import numpy as np
import pytest
from mm_utils import load_data, select_columns, cluster_data, get_param_grid


def test_load_data_missing(tmp_path, capsys):
    missing_file = tmp_path / "missing.csv"
    df = load_data(missing_file)
    captured = capsys.readouterr()
    assert df is None
    assert "was not found" in captured.out


@pytest.fixture
def sample_df():
    return pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})


def test_select_columns(sample_df):
    result = select_columns(sample_df, ["A", "C"])
    assert list(result.columns) == ["A", "C"]


def test_cluster_data_permutation():
    matrix = np.array([[1, 2], [2, 1], [3, 3]])
    order = cluster_data(matrix)
    assert sorted(order) == list(range(matrix.shape[0]))


def test_get_param_grid_non_empty():
    assert get_param_grid("svm")
    assert get_param_grid("logreg")
    assert get_param_grid("rf")
