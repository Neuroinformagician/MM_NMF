#!/usr/bin/env python
# coding: utf-8

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
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

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

def get_column_names(prefixes, suffixes):
    return [f"{prefix}_{suffix}" for prefix in prefixes for suffix in suffixes]

def cluster_data(matrix, method='complete', metric='euclidean'):
    cluster_rows = linkage(matrix, method=method, metric=metric)
    return leaves_list(cluster_rows)

def iqr(group):
    Q1 = group.quantile(0.25)
    Q3 = group.quantile(0.75)
    IQR = Q3 - Q1
    return IQR

def main():
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
    
    numeric_columns = df.select_dtypes(include='number')
    scaler = MinMaxScaler()
    scaled_columns = scaler.fit_transform(numeric_columns)
    df[numeric_columns.columns] = scaled_columns
    
    nmf = NMF(n_components=4, init='nndsvda', max_iter=200, random_state=0)
    
    W = nmf.fit_transform(df)
    H = nmf.components_
    
    feature_names = df.columns
    
    row_order = cluster_data(H)
    col_order = cluster_data(H.T)
    
    H_reordered = H[row_order][:, col_order]
    feature_names_reordered = feature_names[col_order]
    
    module_names = ["module QOL", "module Diplopia", "module Ptosis", "module Systemic"]
    module_names_reordered = [module_names[i] for i in row_order]
    
    plt.figure(figsize=(10, 3), dpi=300)
    sns.heatmap(H_reordered, cmap="Reds", xticklabels=feature_names_reordered, yticklabels=module_names_reordered)
    plt.title("H Matrix")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()
    
    H_df = pd.DataFrame(H, columns=df.columns)
    H_df = H_df.T
    H_df.columns = ["module QOL", "module Diplopia", "module Ptosis", "module Systemic"]
    
    fig = sns.clustermap(H_df,
                   cmap="Reds",
                   xticklabels=True,
                   yticklabels=True,
                   figsize=(5, 8),
                   dendrogram_ratio=(0.1, 0.1),
                   cbar_pos=(-0.12, 0.5, 0.03, 0.2),
                   cbar_kws={'label': 'Coefficient Values'})
    
    plt.setp(fig.ax_heatmap.get_xticklabels(), rotation=45, ha="right") 
    plt.show()
    
    fig.savefig('./fig/fig2.pdf', dpi=1500)
    fig.savefig('./fig/fig2.tiff', dpi=1200)
    
    IDs = df_ori['ID']
    W_with_id = np.column_stack((IDs, W))
    module_names_with_id = ['ID'] + module_names_reordered
    W_with_id_df = pd.DataFrame(W_with_id, columns=module_names_with_id)
    
    module_names_with_id = ['ID', "module QOL", "module Diplopia", "module Ptosis", "module Systemic"]
    
    W_with_id_df = pd.DataFrame(W_with_id, columns=module_names_with_id)
    MM_data = df_ori[["ID", "MMorbetter"]]
    
    W_MM = pd.merge(W_with_id_df, MM_data, on="ID")
    W_MM.index = W_MM['ID']
    W_MM.drop(columns=['ID'], inplace=True)
    
    with open('./out/nmf_model.pkl', 'wb') as file:
        pickle.dump(nmf, file)
    
    with open('./out/W_matrix.pkl', 'wb') as file:
        pickle.dump(W, file)
    
    with open('./out/H_matrix.pkl', 'wb') as file:
        pickle.dump(H, file)
    
    with open('./out/W_with_id_df.pkl', 'wb') as file:
        pickle.dump(W_with_id_df, file)
    
    with open('./out/W_MM.pkl', 'wb') as file:
        pickle.dump(W_MM, file)
    
    grouped_data = W_MM.groupby("MMorbetter")
    MM0 = grouped_data.get_group(0)
    MM1 = grouped_data.get_group(1)
    
    W_MM['Group'] = np.where(W_MM['MMorbetter'] == 1, 'MM1', 'MM0')
    
    MM0_median = MM0.median()
    MM1_median = MM1.median()
    med_df = pd.DataFrame({"MM0_median": MM0_median, "MM1_median": MM1_median}).drop(['MMorbetter'], axis=0)
    
    MM0_iqr = iqr(MM0)
    MM1_iqr = iqr(MM1)
    
    iqr_df = pd.DataFrame({"MM0_median": MM0_median, "MM1_median": MM1_median, "MM0_iqr": MM0_iqr, "MM1_iqr": MM1_iqr}).drop(['MMorbetter'])
    
    your_case = None
    
    long_data = pd.concat([MM0.drop('MMorbetter', axis=1).assign(Group='non MM'),
                           MM1.drop('MMorbetter', axis=1).assign(Group='MM or better')])
    
    MM0_median = MM0.median()
    MM1_median = MM1.median()
    med_df = pd.DataFrame({"MM0_median": MM0_median, "MM1_median": MM1_median}).drop(['MMorbetter'], axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=800)
    
    p_values = [mannwhitneyu(MM0[module], MM1[module])[1] for module in med_df.index]
    corrected_p_values = multipletests(p_values, method="bonferroni")[1]
    
    sns.boxplot(x="variable", y="value", hue="Group", data=pd.melt(long_data, id_vars='Group'), palette=['#1f77b4', '#ff7f0e'], ax=ax, showfliers=False)
    
    for i, p_val in enumerate(corrected_p_values):
        ax.text(i - 0.2, ax.get_ylim()[1]*1.0, f"p={p_val:.2e}", ha="left")
    
    if your_case is not None:
        scatter_pos = 1 if rounded_prediction_proba >= 0.6 else 0
        for i, module in enumerate(med_df.index):
            ax.scatter(i + (scatter_pos - 0.5) * 0.4, your_case[module], color="r", marker="o", s=50, label="Your case" if i == 0 else '', edgecolors='black')
    
    ax.set(ylabel="Feature value")
    ax.set_title("")
    ax.set_xlabel('')
    ax.set_xticklabels(med_df.index)
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1.05))
    plt.subplots_adjust(right=0.85)
    
    sns.despine()
    
    plt.savefig('./fig/fig3.pdf', format='pdf', dpi=1500, transparent=True)
    plt.savefig('./fig/fig3.tiff', format='tiff', dpi=1500, transparent=True)
    
    plt.show()
    
    print("Step 1 completed successfully!")
    print(f"Data shape: {df.shape}")
    print(f"W matrix shape: {W.shape}")
    print(f"H matrix shape: {H.shape}")

if __name__ == "__main__":
    main()