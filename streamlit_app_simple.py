"""
Streamlit MG Prediction App
重症筋無力症（MG）予測システム - シンプル版
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
    layout="centered",
    initial_sidebar_state="collapsed"
)

# カスタムCSS
st.markdown("""
<style>
    /* 基本設定 */
    * {
        box-sizing: border-box;
    }

    /* フォント */
    html, body {
        font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
    }

    /* 背景 */
    .stApp {
        background-color: #f5f5f5;
    }

    /* コンテナ */
    .main .block-container {
        max-width: 600px;
        margin: 0 auto;
        padding: 1rem;
    }

    /* 見出し */
    h1 {
        font-size: 1.8rem;
        font-weight: 600;
        text-align: center;
        color: #333;
        margin-bottom: 1.5rem;
    }

    h2 {
        font-size: 1.4rem;
        font-weight: 600;
        color: #333;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    h3 {
        font-size: 1.1rem;
        font-weight: 600;
        color: #555;
    }

    /* 項目名 */
    .item-name {
        font-size: 1rem;
        font-weight: 600;
        color: #333;
        background: white;
        padding: 10px;
        margin-bottom: 8px;
        border-radius: 8px;
        border-left: 4px solid #007aff;
    }

    /* スコアボタン */
    div.stButton > button {
        width: 100%;
        height: 54px;
        font-size: 1.4rem;
        font-weight: 700;
        border-radius: 8px;
        border: 2px solid #ddd;
        background: white;
        color: #333;
        transition: all 0.2s;
    }

    div.stButton > button:hover {
        background: #f0f0f0;
        border-color: #007aff;
    }

    /* 選択されたボタン */
    div.stButton > button[data-selected="true"] {
        background: #007aff !important;
        border-color: #007aff !important;
        color: white !important;
    }

    /* プライマリボタン */
    div.stButton > button[kind="primary"] {
        background: #007aff;
        color: white;
        border: none;
        height: 48px;
        font-size: 1rem;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #0051d5;
    }

    /* セカンダリボタン */
    div.stButton > button[kind="secondary"] {
        background: #f0f0f0;
        color: #333;
        border: 1px solid #ddd;
        height: 48px;
        font-size: 1rem;
        font-weight: 600;
    }

    /* 合計表示 */
    .total-display {
        background: #007aff;
        color: white;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
        padding: 16px;
        border-radius: 10px;
        margin: 20px 0;
    }

    .total-display.mgc {
        background: #34c759;
    }

    .total-display.mgqol {
        background: #af52de;
    }

    /* スマホ対応 */
    @media screen and (max-width: 500px) {
        .main .block-container {
            padding: 0.5rem;
        }

        h1 {
            font-size: 1.5rem;
        }

        h2 {
            font-size: 1.2rem;
        }

        div.stButton > button {
            height: 50px;
            font-size: 1.3rem;
        }

        .item-name {
            font-size: 0.95rem;
            padding: 8px;
        }

        .total-display {
            font-size: 1.2rem;
            padding: 14px;
        }
    }

    /* 極小画面 */
    @media screen and (max-width: 380px) {
        div.stButton > button {
            height: 46px;
            font-size: 1.2rem;
        }
    }

    /* メトリクス */
    [data-testid="metric-container"] {
        background: white;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }

    /* 横スクロール防止 */
    html, body, .stApp, .main {
        overflow-x: hidden !important;
        max-width: 100vw !important;
    }

    /* カラムの隙間調整 */
    div[data-testid="column"] {
        padding: 0 2px;
    }

    /* 強制的に横並びレイアウト */
    div[data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 4px;
        width: 100% !important;
    }

    /* カラムを横並びに固定 */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        flex: 1 1 0 !important;
        min-width: 0 !important;
        max-width: none !important;
    }

    /* スマホでもカラムを横並び維持 */
    @media screen and (max-width: 640px) {
        div[data-testid="stHorizontalBlock"] {
            display: flex !important;
            flex-direction: row !important;
        }

        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            flex: 1 1 0 !important;
            min-width: 0 !important;
        }
    }

    /* ボタンコンテナの調整 */
    .stButton {
        width: 100%;
    }

    /* ボタンのフォントサイズ調整（小画面） */
    @media screen and (max-width: 380px) {
        div.stButton > button {
            font-size: 1.1rem;
            padding: 0 2px;
        }
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

if 'current_scale' not in st.session_state:
    st.session_state.current_scale = 0

# ============================================================================
# ヘルパー関数
# ============================================================================

def set_score(key, value):
    """スコアを設定"""
    st.session_state.scores[key] = value

def get_score(key):
    """スコアを取得"""
    return st.session_state.scores.get(key, 0)

def reset_all_scores():
    """全スコアをリセット"""
    st.session_state.scores = {}
    st.session_state.prediction_results = None
    st.session_state.current_scale = 0
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
            index = get_score(key)
            total += item['values'][index]
        else:
            total += get_score(key)
    return total

def next_scale():
    """次のスケールに進む"""
    if st.session_state.current_scale < 3:
        st.session_state.current_scale += 1
        st.rerun()

def prev_scale():
    """前のスケールに戻る"""
    if st.session_state.current_scale > 0:
        st.session_state.current_scale -= 1
        st.rerun()

# ============================================================================
# メインUI
# ============================================================================

st.title("MG予測システム")

# 進捗表示
progress_labels = ["MG-ADL", "MG Composite", "MGQOL-15r", "予測結果"]
cols = st.columns(4)
for i, label in enumerate(progress_labels):
    with cols[i]:
        if i == st.session_state.current_scale:
            st.markdown(f"**▶ {label}**")
        elif i < st.session_state.current_scale:
            st.markdown(f"✓ {label}")
        else:
            st.markdown(f"○ {label}")

st.markdown("---")

# ============================================================================
# MG-ADL入力
# ============================================================================

if st.session_state.current_scale == 0:
    st.header("MG-ADL (0-24点)")

    for item in MGADL_ITEMS:
        key = f"adl_{item['key']}"
        current_score = get_score(key)

        # 項目名
        marker = " (→MGC自動反映)" if item['key'] in ['speech', 'chewing', 'swallowing', 'respiration'] else ""
        st.markdown(f'<div class="item-name">{item["name"]}{marker}</div>', unsafe_allow_html=True)

        # スコアボタン（0-3）
        cols = st.columns(4)
        for i in range(4):
            with cols[i]:
                btn_type = "primary" if current_score == i else "secondary"
                if st.button(str(i), key=f"{key}_{i}", type=btn_type, use_container_width=True):
                    set_score(key, i)
                    if item['key'] in ['speech', 'chewing', 'swallowing', 'respiration']:
                        sync_adl_to_mgc()
                    st.rerun()

    # 合計点
    total = calculate_total(MGADL_ITEMS, 'adl')
    st.markdown(f'<div class="total-display">合計: {total}/24点</div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col2:
        if st.button("次へ →", type="primary", use_container_width=True):
            next_scale()

# ============================================================================
# MGC入力
# ============================================================================

elif st.session_state.current_scale == 1:
    st.header("MG Composite (0-50点)")

    for item in MGC_ITEMS:
        key = f"mgc_{item['key']}"
        current_score = get_score(key)

        # 項目名
        marker = " (ADLから自動)" if item['key'] in ['speech', 'chewing', 'swallowing', 'respiration'] else ""
        desc = f" - {item['description']}" if item.get('description') else ""
        st.markdown(f'<div class="item-name">{item["name"]}{marker}{desc}</div>', unsafe_allow_html=True)

        # スコアボタン（実際の値）
        values = item['values']
        cols = st.columns(len(values))
        for i, val in enumerate(values):
            with cols[i]:
                btn_type = "primary" if current_score == i else "secondary"
                if st.button(str(val), key=f"{key}_{i}", type=btn_type, use_container_width=True):
                    set_score(key, i)
                    st.rerun()

    # 合計点
    total = calculate_total(MGC_ITEMS, 'mgc')
    st.markdown(f'<div class="total-display mgc">合計: {total}/50点</div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← 戻る", type="secondary", use_container_width=True):
            prev_scale()
    with col2:
        if st.button("次へ →", type="primary", use_container_width=True):
            next_scale()

# ============================================================================
# MGQOL入力
# ============================================================================

elif st.session_state.current_scale == 2:
    st.header("MGQOL-15r (0-30点)")

    for item in MGQOL_ITEMS:
        key = f"mgqol_{item['key']}"
        current_score = get_score(key)

        # 項目名
        st.markdown(f'<div class="item-name">{item["name"]}</div>', unsafe_allow_html=True)

        # スコアボタン（0-2）
        cols = st.columns(3)
        for i in range(3):
            with cols[i]:
                btn_type = "primary" if current_score == i else "secondary"
                if st.button(str(i), key=f"{key}_{i}", type=btn_type, use_container_width=True):
                    set_score(key, i)
                    st.rerun()

    # 合計点
    total = calculate_total(MGQOL_ITEMS, 'mgqol')
    st.markdown(f'<div class="total-display mgqol">合計: {total}/30点</div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← 戻る", type="secondary", use_container_width=True):
            prev_scale()
    with col2:
        if st.button("予測実行 →", type="primary", use_container_width=True):
            next_scale()

# ============================================================================
# 予測結果
# ============================================================================

elif st.session_state.current_scale == 3:
    st.header("予測結果")

    if st.session_state.prediction_results is None:
        with st.spinner("解析中..."):
            try:
                check_required_files()

                if st.session_state.predictor is None:
                    st.session_state.predictor = MGPredictor()
                    st.session_state.predictor.load_resources()

                predictor = st.session_state.predictor
                patient_df = predictor.convert_ui_scores_to_dataframe(st.session_state.scores)
                module_scores = predictor.transform_to_modules(patient_df)
                results = predictor.predict(module_scores)
                comparison = predictor.get_mm_comparison_data()

                st.session_state.prediction_results = {
                    **results,
                    "comparison": comparison
                }
                st.rerun()

            except Exception as e:
                st.error(f"エラー: {e}")
                st.exception(e)

    if st.session_state.prediction_results is not None:
        results = st.session_state.prediction_results

        # 予測結果
        ensemble_prob = results['ensemble']
        classification = results['classification']
        total_votes = results['total_mm_votes']

        if classification == "MM or better":
            st.success(f"**予測結果: {classification}**")
        else:
            st.error(f"**予測結果: {classification}**")

        st.info(f"確率: {ensemble_prob:.1%} ({total_votes}/3 モデルが予測)")

        # 入力スコア
        st.subheader("入力スコア")
        cols = st.columns(3)
        with cols[0]:
            adl_total = calculate_total(MGADL_ITEMS, 'adl')
            st.metric("MG-ADL", f"{adl_total}/24")
        with cols[1]:
            mgc_total = calculate_total(MGC_ITEMS, 'mgc')
            st.metric("MGC", f"{mgc_total}/50")
        with cols[2]:
            mgqol_total = calculate_total(MGQOL_ITEMS, 'mgqol')
            st.metric("MGQOL", f"{mgqol_total}/30")

        # モデル別予測
        st.subheader("モデル別予測")
        cols = st.columns(3)
        for i, (model_name, vote_info) in enumerate(results['model_votes'].items()):
            with cols[i]:
                prob = vote_info['probability']
                pred = vote_info['prediction']

                if pred == "MM or better":
                    st.success(f"**{model_name}**  \n{prob:.1%} ✓")
                else:
                    st.error(f"**{model_name}**  \n{prob:.1%} ✗")

        # モジュールスコア
        st.subheader("モジュールスコア")
        module_df = results['module_scores'].copy()
        module_df.columns = [col.replace('module ', '').title() for col in module_df.columns]

        cols = st.columns(4)
        for i, (col_name, value) in enumerate(module_df.iloc[0].items()):
            with cols[i]:
                st.metric(col_name, f"{value:.4f}")

        # レーダーチャート
        st.subheader("比較チャート")
        comparison = results['comparison']
        patient_scores = results['module_scores'].values[0]

        fig = create_radar_chart(
            patient_scores=patient_scores,
            mm_avg=comparison['mm_avg'],
            non_mm_avg=comparison['non_mm_avg'],
            module_names=comparison['module_names'],
            ensemble_prob=ensemble_prob
        )
        st.pyplot(fig, use_container_width=True)

        # 詳細
        with st.expander("詳細データ"):
            st.write("**モデル別詳細:**")
            for model_name, vote_info in results['model_votes'].items():
                st.write(f"- {model_name}: {vote_info['probability']:.4f} (カットオフ: {vote_info['cutoff']:.4f})")

            st.write("\n**5-fold予測値:**")
            for model_name, probs in results['predictions'].items():
                st.write(f"- {model_name}: {probs}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← 入力に戻る", type="secondary", use_container_width=True):
            prev_scale()

# ============================================================================
# リセットボタン
# ============================================================================

st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("全データリセット", use_container_width=True):
        reset_all_scores()

# ============================================================================
# フッター
# ============================================================================

st.markdown("""
<div style='text-align: center; margin-top: 40px; color: #999; font-size: 0.9rem;'>
    MG予測システム v3.2
</div>
""", unsafe_allow_html=True)