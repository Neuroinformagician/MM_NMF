"""
Streamlit MG Prediction App - Visualizer
可視化ロジック: Matplotlibによるレーダーチャート
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import matplotlib.font_manager as fm


def create_radar_chart(patient_scores, mm_avg, non_mm_avg, module_names, ensemble_prob):
    """
    レーダーチャートを作成（matplotlib使用）

    Parameters
    ----------
    patient_scores : np.array
        患者のモジュールスコア
    mm_avg : np.array
        MM群の平均スコア
    non_mm_avg : np.array
        non-MM群の平均スコア
    module_names : list
        モジュール名のリスト
    ensemble_prob : float
        アンサンブル予測確率

    Returns
    -------
    matplotlib.figure.Figure
        レーダーチャート
    """
    # データ保護（コピー）
    patient_scores = patient_scores.copy()
    mm_avg = mm_avg.copy()
    non_mm_avg = non_mm_avg.copy()

    # モジュール名を整形
    labels = [name.replace('module ', '').title() for name in module_names]

    # データの最大値を取得
    max_val = max(
        np.max(patient_scores),
        np.max(mm_avg),
        np.max(non_mm_avg)
    )
    display_range = max(0.15, max_val * 1.1)

    # 角度を計算
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # データを閉じた形にする
    patient_closed = np.concatenate([patient_scores, [patient_scores[0]]])
    mm_closed = np.concatenate([mm_avg, [mm_avg[0]]])
    non_mm_closed = np.concatenate([non_mm_avg, [non_mm_avg[0]]])
    angles_closed = angles + [angles[0]]

    # Figure作成（大きめに）
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # 背景を白に
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # non-MM群（薄い赤系）
    ax.plot(angles_closed, non_mm_closed, 'o-', linewidth=3,
            color='#FF6B6B', alpha=0.6, label='non MM (mean)', markersize=8)
    ax.fill(angles_closed, non_mm_closed, alpha=0.15, color='#FF6B6B')

    # MM群（薄い青緑系）
    ax.plot(angles_closed, mm_closed, 'o-', linewidth=3,
            color='#4ECDC4', alpha=0.6, label='MM or better (mean)', markersize=8)
    ax.fill(angles_closed, mm_closed, alpha=0.15, color='#4ECDC4')

    # 患者データ（黄色、強調）
    ax.plot(angles_closed, patient_closed, 'o-', linewidth=5,
            color='#FFC107', label='This patient', markersize=16,
            markeredgecolor='white', markeredgewidth=3)
    ax.fill(angles_closed, patient_closed, alpha=0.3, color='#FFC107')

    # 軸ラベル
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, size=20, weight='bold')

    # 放射軸の設定
    ax.set_ylim(0, display_range)
    ax.set_yticks([0.05, 0.10, 0.15])
    ax.set_yticklabels(['0.05', '0.10', '0.15'], size=16, color='gray')
    ax.grid(True, linewidth=1.5, color='lightgray', alpha=0.7)

    # 凡例
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              ncol=3, fontsize=16, frameon=True, fancybox=True,
              edgecolor='black', framealpha=0.95)

    plt.tight_layout()

    return fig


def create_module_comparison_table(patient_scores, mm_avg, non_mm_avg, module_names):
    """
    モジュールスコア比較テーブルを作成

    Parameters
    ----------
    patient_scores : np.array
        患者のモジュールスコア
    mm_avg : np.array
        MM群の平均スコア
    non_mm_avg : np.array
        non-MM群の平均スコア
    module_names : list
        モジュール名のリスト

    Returns
    -------
    pd.DataFrame
        比較テーブル
    """
    import pandas as pd

    comparison_data = {
        'Module': [name.replace('module ', '').title() for name in module_names],
        'non MM (mean)': [f"{x:.4f}" for x in non_mm_avg],
        'MM or better (mean)': [f"{x:.4f}" for x in mm_avg],
        'This patient': [f"{x:.4f}" for x in patient_scores],
        'Diff vs MM': [f"{x - y:.4f}" for x, y in zip(patient_scores, mm_avg)]
    }

    df = pd.DataFrame(comparison_data)
    return df


def create_module_assessment_text(patient_scores, mm_avg, non_mm_avg, module_names):
    """
    モジュール別評価テキストを作成

    Parameters
    ----------
    patient_scores : np.array
        患者のモジュールスコア
    mm_avg : np.array
        MM群の平均スコア
    non_mm_avg : np.array
        non-MM群の平均スコア
    module_names : list
        モジュール名のリスト

    Returns
    -------
    list of dict
        評価結果のリスト
        [{"module": "QOL", "score": 0.0391, "status": "better", "icon": "✓"}, ...]
    """
    assessments = []

    for i, module in enumerate(module_names):
        module_short = module.replace('module ', '').title()
        patient_val = patient_scores[i]
        mm_val = mm_avg[i]
        non_mm_val = non_mm_avg[i]

        if patient_val < non_mm_val:
            status = "Better than non-MM"
            icon = "✓"
            color = "green"
        elif patient_val > mm_val:
            status = "Worse than MM"
            icon = "⚠"
            color = "red"
        else:
            status = "Between groups"
            icon = "○"
            color = "gray"

        assessments.append({
            "module": module_short,
            "score": patient_val,
            "status": status,
            "icon": icon,
            "color": color
        })

    return assessments
