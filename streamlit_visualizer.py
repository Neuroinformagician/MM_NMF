"""
Streamlit MG Prediction App - Visualizer
可視化ロジック: Plotlyによるインタラクティブレーダーチャート
"""

import plotly.graph_objects as go
import numpy as np


def create_radar_chart(patient_scores, mm_avg, non_mm_avg, module_names, ensemble_prob):
    """
    レーダーチャートを作成

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
    plotly.graph_objects.Figure
        レーダーチャート
    """
    # データ保護（コピー）
    patient_scores = patient_scores.copy()
    mm_avg = mm_avg.copy()
    non_mm_avg = non_mm_avg.copy()

    # モジュール名を整形
    labels = [name.replace('module ', '').title() for name in module_names]

    # 閉じた形にする（最初の値を最後に追加）
    patient_closed = np.concatenate([patient_scores, [patient_scores[0]]])
    mm_closed = np.concatenate([mm_avg, [mm_avg[0]]])
    non_mm_closed = np.concatenate([non_mm_avg, [non_mm_avg[0]]])
    labels_closed = labels + [labels[0]]

    # Figureを作成
    fig = go.Figure()

    # non-MM群（薄い赤系）
    fig.add_trace(go.Scatterpolar(
        r=non_mm_closed,
        theta=labels_closed,
        fill='toself',
        fillcolor='rgba(255, 107, 107, 0.08)',
        line=dict(color='rgba(255, 107, 107, 0.4)', width=2),
        marker=dict(size=6, color='rgba(255, 107, 107, 0.4)'),
        name='non MM (mean)',
        hovertemplate='<b>%{theta}</b><br>Score: %{r:.4f}<extra></extra>'
    ))

    # MM群（薄い青緑系）
    fig.add_trace(go.Scatterpolar(
        r=mm_closed,
        theta=labels_closed,
        fill='toself',
        fillcolor='rgba(78, 205, 196, 0.08)',
        line=dict(color='rgba(78, 205, 196, 0.4)', width=2),
        marker=dict(size=6, color='rgba(78, 205, 196, 0.4)'),
        name='MM or better (mean)',
        hovertemplate='<b>%{theta}</b><br>Score: %{r:.4f}<extra></extra>'
    ))

    # 患者データ（黄色、強調）
    fig.add_trace(go.Scatterpolar(
        r=patient_closed,
        theta=labels_closed,
        fill='toself',
        fillcolor='rgba(255, 193, 7, 0.3)',
        line=dict(color='#FFC107', width=4),
        marker=dict(size=14, color='#FFC107', line=dict(color='white', width=2)),
        name='This patient',
        hovertemplate='<b>%{theta}</b><br>Score: %{r:.4f}<extra></extra>'
    ))

    # データの最大値を取得（はみ出し対策）
    max_val = max(
        np.max(patient_closed),
        np.max(mm_closed),
        np.max(non_mm_closed)
    )
    # 円の範囲を0.15だが、データが超える場合は自動拡張
    display_range = max(0.15, max_val * 1.1)

    # レイアウト設定
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, display_range],
                tickvals=[0.05, 0.10, 0.15],
                ticktext=['0.05', '0.10', '0.15'],
                tickfont=dict(size=13, color='gray'),
                showline=False,
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1
            ),
            angularaxis=dict(
                tickfont=dict(size=15, color='black')
            ),
            bgcolor='white'
        ),
        showlegend=True,
        legend=dict(
            x=1.05,
            y=1.0,
            font=dict(size=12),
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='black',
            borderwidth=1
        ),
        height=500,
        margin=dict(l=50, r=150, t=50, b=50)
    )

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
