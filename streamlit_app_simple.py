"""
Streamlit MG Prediction App
重症筋無力症（MG）予測システム - シンプルボタン入力版
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from streamlit_config import (
    MGADL_ITEMS, MGC_ITEMS, MGQOL_ITEMS,
    check_required_files
)
from streamlit_predictor import MGPredictor
from streamlit_visualizer import create_radar_chart

# ============================================================================
# ページ設定
# ============================================================================

st.set_page_config(
    page_title="MG予測システム",
    page_icon="🏥",
    layout="wide"
)

# カスタムCSSでボタンにタップエフェクトを追加
st.markdown("""
<style>
    /* ボタンにホバー＆アクティブエフェクト */
    div.stButton > button {
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
    }

    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    div.stButton > button:active {
        transform: scale(0.95);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    /* ボタンクリック時の波紋エフェクト */
    @keyframes ripple {
        0% {
            transform: scale(0);
            opacity: 0.6;
        }
        100% {
            transform: scale(4);
            opacity: 0;
        }
    }

    div.stButton > button:active::after {
        content: "";
        position: absolute;
        top: 50%;
        left: 50%;
        width: 20px;
        height: 20px;
        background: rgba(255, 255, 255, 0.5);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        animation: ripple 0.6s ease-out;
    }

    /* プライマリボタン（予測実行）を目立たせる */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-weight: 600;
        font-size: 1.1em;
    }

    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# セッション状態の初期化
# ============================================================================

if 'scores' not in st.session_state:
    st.session_state.scores = {}

if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

if 'predictor' not in st.session_state:
    st.session_state.predictor = None

# ============================================================================
# ヘルパー関数
# ============================================================================

def increment_score(key, max_val):
    """スコアを1増やす（最大値でループ）"""
    if key not in st.session_state.scores:
        st.session_state.scores[key] = 0
    st.session_state.scores[key] = (st.session_state.scores[key] + 1) % (max_val + 1)

def get_score(key):
    """スコアを取得（デフォルト0）"""
    return st.session_state.scores.get(key, 0)

def reset_all_scores():
    """全スコアをリセット"""
    st.session_state.scores = {}
    st.session_state.prediction_results = None
    st.session_state.current_scale = 0  # MG-ADLに戻る
    st.rerun()

def sync_adl_to_mgc():
    """ADLからMGCへの自動同期"""
    adl_to_mgc_map = {
        "adl_speech": "mgc_speech",
        "adl_chewing": "mgc_chewing",
        "adl_swallowing": "mgc_swallowing",
        "adl_respiration": "mgc_respiration"
    }

    for adl_key, mgc_key in adl_to_mgc_map.items():
        if adl_key in st.session_state.scores:
            st.session_state.scores[mgc_key] = st.session_state.scores[adl_key]

def calculate_total(items, prefix):
    """合計点を計算"""
    total = 0
    for item in items:
        key = f"{prefix}_{item['key']}"
        if prefix == "mgc":
            # MGCは実際の点数配列から取得
            index = get_score(key)
            total += item['values'][index]
        else:
            total += get_score(key)
    return total

# ============================================================================
# ナビゲーション用のセッション状態
# ============================================================================

if 'current_scale' not in st.session_state:
    st.session_state.current_scale = 0  # 0: ADL, 1: MGC, 2: MGQOL

def next_scale():
    """次のスケールに進む"""
    if st.session_state.current_scale < 2:
        st.session_state.current_scale += 1
        st.rerun()

def prev_scale():
    """前のスケールに戻る"""
    if st.session_state.current_scale > 0:
        st.session_state.current_scale -= 1
        st.rerun()

def go_to_scale(scale_index):
    """指定のスケールに移動"""
    st.session_state.current_scale = scale_index
    st.rerun()

# ============================================================================
# メインUI
# ============================================================================

st.title("MG予測システム")

# 進捗インジケーター
progress_labels = ["MG-ADL", "MG Composite", "MGQOL-15r"]
cols = st.columns(3)
for i, label in enumerate(progress_labels):
    with cols[i]:
        if i == st.session_state.current_scale:
            st.markdown(f"**:blue[{label}]** ⬅️")
        elif i < st.session_state.current_scale:
            st.markdown(f"✅ {label}")
        else:
            st.markdown(f"⚪ {label}")

st.markdown("---")

# ============================================================================
# MG-ADL入力
# ============================================================================

# MG-ADL入力
if st.session_state.current_scale == 0:
    st.header("MG-ADL (0-24点)")

    cols = st.columns(2)
    for i, item in enumerate(MGADL_ITEMS):
        key = f"adl_{item['key']}"
        max_val = len(item['options']) - 1  # 0-3なので、4つのオプション - 1

        with cols[i % 2]:
            col1, col2 = st.columns([3, 1])
            with col1:
                # 自動反映される項目には印をつける
                marker = " ⚡" if item['key'] in ['speech', 'chewing', 'swallowing', 'respiration'] else ""
                st.markdown(f"**{item['name']}{marker}**")
            with col2:
                if st.button(f"{get_score(key)}", key=f"btn_{key}", use_container_width=True):
                    increment_score(key, max_val)
                    # 自動的にMGCに反映
                    sync_adl_to_mgc()
                    st.rerun()

    st.markdown(f"**合計: {calculate_total(MGADL_ITEMS, 'adl')}/24**")
    st.caption("⚡印の項目はMG Compositeに自動反映されます")

    st.markdown("---")

    # ナビゲーションボタン
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        if st.button("MGCへ →", type="primary", use_container_width=True):
            next_scale()

# ============================================================================
# MGC入力
# ============================================================================

# MGC入力
elif st.session_state.current_scale == 1:
    st.header("MG Composite (0-50点)")

    cols = st.columns(2)
    for i, item in enumerate(MGC_ITEMS):
        key = f"mgc_{item['key']}"
        max_val = len(item['values']) - 1  # values配列のインデックス最大値

        with cols[i % 2]:
            col1, col2 = st.columns([3, 1])
            with col1:
                marker = " ⚡" if item['key'] in ['speech', 'chewing', 'swallowing', 'respiration'] else ""
                st.markdown(f"**{item['name']}{marker}**")
            with col2:
                # 現在のインデックスを取得して、実際の点数を表示
                current_index = get_score(key)
                actual_score = item['values'][current_index]
                if st.button(f"{actual_score}", key=f"btn_{key}", use_container_width=True):
                    increment_score(key, max_val)
                    st.rerun()

    st.markdown(f"**合計: {calculate_total(MGC_ITEMS, 'mgc')}/50**")
    st.caption("⚡印の項目はMG-ADLから自動反映されます")

    st.markdown("---")

    # ナビゲーションボタン
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← ADLへ戻る", use_container_width=True):
            prev_scale()
    with col3:
        if st.button("MGQOLへ →", type="primary", use_container_width=True):
            next_scale()

# ============================================================================
# MGQOL入力
# ============================================================================

# MGQOL入力
elif st.session_state.current_scale == 2:
    st.header("MGQOL-15r (0-30点)")

    cols = st.columns(2)
    for i, item in enumerate(MGQOL_ITEMS):
        key = f"mgqol_{item['key']}"
        max_val = 2  # MGQOLは全て0-2点

        with cols[i % 2]:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{item['name']}**")
            with col2:
                if st.button(f"{get_score(key)}", key=f"btn_{key}", use_container_width=True):
                    increment_score(key, max_val)
                    st.rerun()

    st.markdown(f"**合計: {calculate_total(MGQOL_ITEMS, 'mgqol')}/30**")

    st.markdown("---")

    # ナビゲーションと予測ボタン
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← MGCへ戻る", use_container_width=True):
            prev_scale()
    with col3:
        predict_btn = st.button("予測実行", type="primary", use_container_width=True)

# ============================================================================
# リセットボタン（全画面共通）
# ============================================================================

st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    reset_btn = st.button("🔄 全データリセット", use_container_width=True)

if reset_btn:
    reset_all_scores()

# ============================================================================
# 予測実行
# ============================================================================

if 'predict_btn' in locals() and predict_btn:
    with st.spinner("予測中..."):
        try:
            check_required_files()

            if st.session_state.predictor is None:
                st.session_state.predictor = MGPredictor()
                st.session_state.predictor.load_resources()

            predictor = st.session_state.predictor

            # UIスコアをDataFrameに変換
            patient_df = predictor.convert_ui_scores_to_dataframe(st.session_state.scores)

            # モジュールスコアに変換
            module_scores = predictor.transform_to_modules(patient_df)

            # 予測実行
            results = predictor.predict(module_scores)

            # MM/non-MM比較データ取得
            comparison = predictor.get_mm_comparison_data()

            # 結果をセッションに保存
            st.session_state.prediction_results = {
                **results,
                "comparison": comparison
            }

            st.success("✅ 予測完了！")

        except FileNotFoundError as e:
            st.error(f"❌ ファイルエラー: {e}")
        except Exception as e:
            st.error(f"❌ 予測エラー: {e}")
            st.exception(e)

# ============================================================================
# 結果表示
# ============================================================================

if st.session_state.prediction_results is not None:
    results = st.session_state.prediction_results

    st.markdown("---")

    # 予測結果表示（目立つように）
    ensemble_prob = results['ensemble']
    classification = results['classification']
    total_votes = results['total_mm_votes']
    total_models = results['total_models']

    # 確率に応じて色を変更
    if classification == "MM or better":
        color = "#4CAF50"  # 緑
    else:
        color = "#F44336"  # 赤

    st.markdown(f"""
    <div style='background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;'>
        <h2 style='color: white; margin: 0;'>予測結果</h2>
        <h1 style='color: white; margin: 10px 0; font-size: 48px;'>{classification}</h1>
        <h3 style='color: white; margin: 0;'>{total_votes}/3 モデルがMM or better</h3>
        <p style='color: white; margin: 10px 0; font-size: 14px;'>アンサンブル確率: {ensemble_prob:.1%}</p>
    </div>
    """, unsafe_allow_html=True)

    # 入力スコア合計の表示
    st.markdown("### 入力スコア")
    score_cols = st.columns(3)
    with score_cols[0]:
        adl_total = calculate_total(MGADL_ITEMS, 'adl')
        st.metric("MG-ADL", f"{adl_total}/24")
    with score_cols[1]:
        mgc_total = calculate_total(MGC_ITEMS, 'mgc')
        st.metric("MG Composite", f"{mgc_total}/50")
    with score_cols[2]:
        mgqol_total = calculate_total(MGQOL_ITEMS, 'mgqol')
        st.metric("MGQOL-15r", f"{mgqol_total}/30")

    # 各モデルの予測詳細
    st.markdown("### 各モデルの予測")

    # 詳細情報（折りたたみ）
    with st.expander("📊 詳細"):
        st.write("**各モデルの詳細データ:**")
        for model_name, vote_info in results['model_votes'].items():
            st.write(f"**{model_name}**")
            st.write(f"  - 確率: {vote_info['probability']:.6f}")
            st.write(f"  - カットオフ: {vote_info['cutoff']:.6f}")
            st.write(f"  - 判定: {vote_info['prediction']}")

        st.write("\n**5-fold予測値:**")
        for model_name, probs in results['predictions'].items():
            st.write(f"**{model_name}**: {[f'{p:.4f}' for p in probs]}")

    cols = st.columns(3)
    for i, (model_name, vote_info) in enumerate(results['model_votes'].items()):
        with cols[i]:
            prob = vote_info['probability']
            cutoff = vote_info['cutoff']
            pred = vote_info['prediction']

            # カラー: MM or better なら緑、そうでなければ赤
            if pred == "MM or better":
                badge_color = "#4CAF50"
                icon = "✓"
            else:
                badge_color = "#F44336"
                icon = "✗"

            st.markdown(f"""
            <div style='padding: 15px; border: 2px solid {badge_color}; border-radius: 10px; text-align: center;'>
                <div style='font-weight: bold; font-size: 14px; margin-bottom: 5px;'>{model_name}</div>
                <div style='font-size: 24px; font-weight: bold; color: {badge_color};'>{prob:.1%}</div>
                <div style='font-size: 20px; margin-top: 5px;'>{icon}</div>
            </div>
            """, unsafe_allow_html=True)

    # モジュールスコア表示
    st.markdown("### モジュールスコア")

    module_df = results['module_scores'].copy()
    module_df.columns = [col.replace('module ', '').title() for col in module_df.columns]

    cols = st.columns(4)
    for i, (col_name, value) in enumerate(module_df.iloc[0].items()):
        with cols[i]:
            st.metric(col_name, f"{value:.4f}")

    # レーダーチャート
    st.markdown("### 比較チャート")

    comparison = results['comparison']
    patient_scores = results['module_scores'].values[0]

    fig = create_radar_chart(
        patient_scores=patient_scores,
        mm_avg=comparison['mm_avg'],
        non_mm_avg=comparison['non_mm_avg'],
        module_names=comparison['module_names'],
        ensemble_prob=ensemble_prob
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("各スケールのタブから患者データを入力し、「予測実行」ボタンをクリックしてください。")

# ============================================================================
# フッター
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>MG予測システム v1.0 | 重症筋無力症 MM予測</small>
</div>
""", unsafe_allow_html=True)
