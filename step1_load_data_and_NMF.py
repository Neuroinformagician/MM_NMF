# step1. load data and NMF

import pickle
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage

import matplotlib.pyplot as plt
from matplotlib import font_manager

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (auc, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, roc_curve)
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF

seed = None

def load_data(filepath):
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
    return df[column_names]

df_names = [
    "MGC_ptosis", "MGC_diplopia", "MGC_eyelid_closure", "MGC_speech", "MGC_chewing",
    "MGC_swallowing", "MGC_respiration", "MGC_neck", "MGC_upper_limb", "MGC_lower_limb",
    "MGADL_speech", "MGADL_chewing", "MGADL_swallowing", "MGADL_respiration",
    "MGADL_toothbrushing", "MGADL_getting_up", "MGADL_diplopia", "MGADL_ptosis",
    "MGQOL1_dissatisfaction", "MGQOL2_seeing", "MGQOL3_eating",
    "MGQOL4_social_activity_restriction", "MGQOL5_hobby_entertainment", "MGQOL6_family_role",
    "MGQOL7_behavior_modification", "MGQOL8_work_impact", "MGQOL9_speaking",
    "MGQOL10_driving", "MGQOL11_feeling_down", "MGQOL12_walking", "MGQOL13_quick_action",
    "MGQOL14_mental_crushing", "MGQOL15_dressing"
]

config = {
    "filepath": "./data/df_4th.csv",
}

df_ori = load_data(config["filepath"])

if df_ori is not None:
    df = select_columns(df_ori, df_names)

numeric_columns = df.select_dtypes(include="number")
scaler = MinMaxScaler()
scaled_columns = scaler.fit_transform(numeric_columns)

scaled_df = df.copy()
scaled_df[numeric_columns.columns] = scaled_columns

nmf = NMF(n_components=4, init="nndsvd", max_iter=200, random_state=0)

W = nmf.fit_transform(scaled_df)
H = nmf.components_


def cluster_data(matrix, method="complete", metric="euclidean"):
    cluster_rows = linkage(matrix, method=method, metric=metric)
    return leaves_list(cluster_rows)

feature_names = scaled_df.columns

row_order = cluster_data(H)
col_order = cluster_data(H.T)


H_reordered = H[row_order][:, col_order]
feature_names_reordered = feature_names[col_order]

module_names = ["module QOL", "module Diplopia", "module Ptosis", "module Systemic"]
module_names_reordered = [module_names[i] for i in row_order]

IDs = df_ori["ID"]
W_with_id = np.column_stack((IDs, W))
module_names_with_id = ["ID"] + module_names_reordered
W_with_id_df = pd.DataFrame(W_with_id, columns=module_names_with_id)
MM_data = df_ori[["ID", "MMorbetter"]]

W_MM = pd.merge(W_with_id_df, MM_data, on="ID")
W_MM.index = W_MM["ID"]
W_MM.drop(columns=["ID"], inplace=True)

with open("./out/nmf_model.pkl", "wb") as file:
    pickle.dump(nmf, file)

with open("./out/W_matrix.pkl", "wb") as file:
    pickle.dump(W, file)

with open("./out/H_matrix.pkl", "wb") as file:
    pickle.dump(H, file)

with open("./out/W_with_id_df.pkl", "wb") as file:
    pickle.dump(W_with_id_df, file)

with open("./out/W_MM.pkl", "wb") as file:
    pickle.dump(W_MM, file)
