#!/usr/bin/env python
# coding: utf-8

"""
MM群とnon-MM群の統計値を計算して保存
"""

import pickle
import pandas as pd
import numpy as np

def calculate_group_statistics():
    """MM群とnon-MM群の平均値と標準偏差を計算"""
    
    # W_MMデータを読み込み
    with open('./out/W_MM.pkl', 'rb') as f:
        W_MM = pickle.load(f)
    
    # MMorbetterで群分け
    MM0 = W_MM[W_MM['MMorbetter'] == 0]  # non-MM群
    MM1 = W_MM[W_MM['MMorbetter'] == 1]  # MM群
    
    # 各モジュールの統計値を計算
    modules = ["module QOL", "module Diplopia", "module Ptosis", "module Systemic"]
    
    stats = {
        'MM_mean': {},
        'MM_std': {},
        'nonMM_mean': {},
        'nonMM_std': {},
        'MM_median': {},
        'nonMM_median': {},
        'MM_q25': {},
        'MM_q75': {},
        'nonMM_q25': {},
        'nonMM_q75': {}
    }
    
    for module in modules:
        # MM群
        stats['MM_mean'][module] = MM1[module].mean()
        stats['MM_std'][module] = MM1[module].std()
        stats['MM_median'][module] = MM1[module].median()
        stats['MM_q25'][module] = MM1[module].quantile(0.25)
        stats['MM_q75'][module] = MM1[module].quantile(0.75)
        
        # non-MM群
        stats['nonMM_mean'][module] = MM0[module].mean()
        stats['nonMM_std'][module] = MM0[module].std()
        stats['nonMM_median'][module] = MM0[module].median()
        stats['nonMM_q25'][module] = MM0[module].quantile(0.25)
        stats['nonMM_q75'][module] = MM0[module].quantile(0.75)
    
    # 保存
    with open('./out/group_statistics.pkl', 'wb') as f:
        pickle.dump(stats, f)
    
    print("群統計値を計算しました:")
    print(f"MM群 (n={len(MM1)}):")
    for module in modules:
        print(f"  {module}: {stats['MM_mean'][module]:.3f} ± {stats['MM_std'][module]:.3f}")
    
    print(f"\nnon-MM群 (n={len(MM0)}):")
    for module in modules:
        print(f"  {module}: {stats['nonMM_mean'][module]:.3f} ± {stats['nonMM_std'][module]:.3f}")
    
    return stats

if __name__ == "__main__":
    calculate_group_statistics()