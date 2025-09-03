#!/usr/bin/env python
# coding: utf-8

"""
学習時のスケーラー、列順、閾値を保存するスクリプト
step1とstep2の後に実行してください
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 特徴量の列順を定義（学習時と同じ）
FEATURE_COLS = [
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

# データを読み込んでスケーラーを学習
df_ori = pd.read_csv('./data/df_4th.csv', index_col=0)
df = df_ori[FEATURE_COLS]

# MinMaxScalerを学習
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# 保存
# 1. スケーラー
with open('./out/minmax_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ MinMaxScaler saved to ./out/minmax_scaler.pkl")

# 2. 列順
with open('./out/feature_cols.pkl', 'wb') as f:
    pickle.dump(FEATURE_COLS, f)
print("✅ Feature columns order saved to ./out/feature_cols.pkl")

# 3. 閾値（オプション - デフォルトは0.5、最適化した場合はここで設定）
thresholds = {
    "ensemble": 0.5,  # アンサンブルの閾値
    "LR": 0.5,        # ロジスティック回帰
    "SVM": 0.5,       # SVM
    "RF": 0.5,        # ランダムフォレスト
    "NB": 0.5         # ナイーブベイズ
}

with open('./out/thresholds.pkl', 'wb') as f:
    pickle.dump(thresholds, f)
print("✅ Thresholds saved to ./out/thresholds.pkl")

print("\n📊 Scaler statistics:")
print(f"  - Min values: {scaler.data_min_[:5]}... (showing first 5)")
print(f"  - Max values: {scaler.data_max_[:5]}... (showing first 5)")
print(f"  - Scale: {scaler.scale_[:5]}... (showing first 5)")
print(f"  - Feature range: {scaler.feature_range}")

print("\n✅ All training artifacts saved successfully!")