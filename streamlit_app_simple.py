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

# カスタムCSS - iOS風デザイン、確実にモバイル対応
st.markdown("""
<style>
    /* iOS風デザイン */
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;500;600;700&display=swap');

    * {
        box-sizing: border-box;
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
        background-color: #f2f2f7;
    }

    /* 横スクロールを完全に防止 */
    html, body, .stApp, .main {
        overflow-x: hidden !important;
        max-width: 100vw !important;
    }

    /* 全体のコンテナ */
    .main .block-container {
        max-width: 800px;
        padding: 1rem 0.5rem;
    }

    /* ヘッダー */
    h1, h2, h3 {
        font-weight: 600;
        color: #1c1c1e;
    }

    /* ラジオボタングループを横並びに、iOS風ボタンスタイル */
    div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        gap: 3px !important;
        width: 100% !important;
        max-width: 100% !important;
        flex-wrap: nowrap !important;
    }

    /* 各ラジオボタンオプション */
    div[role="radiogroup"] > label {
        flex: 1 1 0 !important;
        min-width: 0 !important;
        max-width: 25% !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* ラジオボタンの見た目をボタン風に */
    div[role="radiogroup"] > label > div {
        width: 100% !important;
        padding: 10px 4px !important;
        text-align: center !important;
        border: 2px solid #e5e5ea !important;
        border-radius: 8px !important;
        background-color: white !important;
        color: #1c1c1e !important;
        font-weight: 600 !important;
        font-size: 1.1em !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        height: 50px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    /* 選択されたラジオボタン（青色） */
    div[role="radiogroup"] > label > div[data-checked="true"] {
        background-color: #007aff !important;
        border-color: #007aff !important;
        color: white !important;
    }

    /* ラジオボタンの丸アイコンを非表示 */
    div[role="radiogroup"] input[type="radio"] {
        display: none !important;
    }

    /* ホバーエフェクト */
    div[role="radiogroup"] > label > div:hover {
        background-color: #f2f2f7 !important;
        border-color: #007aff !important;
    }

    div[role="radiogroup"] > label > div[data-checked="true"]:hover {
        background-color: #0062cc !important;
    }

    /* 項目ラベル */
    .item-label {
        font-weight: 600;
        font-size: 1.1em;
        color: #1c1c1e;
        margin-top: 15px;
        margin-bottom: 8px;
        padding: 10px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    /* 合計点の表示 */
    .total-score {
        font-size: 1.5em;
        font-weight: 700;
        text-align: center;
        padding: 20px;
        background-color: white;
        border-radius: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }

    /* ナビゲーションボタン */
    div.stButton > button {
        min-height: 60px;
        font-size: 1.2em;
        font-weight: 600;
        border-radius: 10px;
    }

    /* プライマリボタン */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
    }

    /* モバイル対応 */
    @media screen and (max-width: 768px) {
        .main .block-container {
            padding: 0.5rem 0.25rem;
        }

        div[role="radiogroup"] > label > div {
            font-size: 1em !important;
            padding: 8px 2px !important;
            height: 48px !important;
            border-radius: 6px !important;
        }

        div[role="radiogroup"] {
            gap: 2px !important;
        }

        .item-label {
            font-size: 1em;
            padding: 8px;
            margin-top: 12px;
            margin-bottom: 6px;
        }

        .total-score {
            font-size: 1.3em;
            padding: 15px;
        }

        div.stButton > button {
            min-height: 55px;
            font-size: 1.1em;
        }
    }

    /* 超小型デバイス対応 */
    @media screen and (max-width: 400px) {
        .main .block-container {
            padding: 0.5rem 0.2rem;
        }

        div[role="radiogroup"] > label > div {
            font-size: 0.95em !important;
            padding: 6px 1px !important;
            height: 45px !important;
            border-width: 1px !important;
            border-radius: 5px !important;
        }

        div[role="radiogroup"] {
            gap: 1px !important;
        }

        .item-label {
            font-size: 0.95em;
            padding: 6px;
            margin-top: 10px;
            margin-bottom: 5px;
        }

        .total-score {
            font-size: 1.2em;
            padding: 12px;
        }
    }

    /* 極小デバイス対応 */
    @media screen and (max-width: 350px) {
        div[role="radiogroup"] > label > div {
            font-size: 0.9em !important;
            padding: 4px 0px !important;
            height: 42px !important;
        }

        .item-label {
            font-size: 0.9em;
            padding: 5px;
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

# ============================================================================
# ヘルパー関数
# ============================================================================

def get_score(key):
    """スコアを取得（デフォルト0）"""
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

# ============================================================================
# ナビゲーション用のセッション状態
# ============================================================================

if 'current_scale' not in st.session_state:
    st.session_state.current_scale = 0

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

st.title("🏥 MG予測システム")

# 進捗インジケーター
progress_labels = ["MG-ADL", "MG Composite", "MGQOL-15r", "予測結果"]
cols = st.columns(4)
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

if st.session_state.current_scale == 0:
    st.header("MG-ADL (0-24点)")
    st.caption("各項目について、該当するスコアを選択してください")

    for item in MGADL_ITEMS:
        key = f"adl_{item['key']}"
        current_score = get_score(key)

        marker = " ⚡" if item['key'] in ['speech', 'chewing', 'swallowing', 'respiration'] else ""
        st.markdown(f'<div class="item-label">{item["name"]}{marker}</div>', unsafe_allow_html=True)

        # st.radioを使って確実に横並び
        selected = st.radio(
            label=item['name'],
            options=list(range(len(item['options']))),
            index=current_score,
            horizontal=True,
            key=key,
            label_visibility="collapsed"
        )

        # スコアを保存
        st.session_state.scores[key] = selected

        # 自動的にMGCに反映
        if item['key'] in ['speech', 'chewing', 'swallowing', 'respiration']:
            sync_adl_to_mgc()

    total = calculate_total(MGADL_ITEMS, 'adl')
    st.markdown(f'<div class="total-score">合計: {total}/24点</div>', unsafe_allow_html=True)
    st.caption("⚡印の項目はMG Compositeに自動反映されます")

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        if st.button("MGCへ →", type="primary", use_container_width=True):
            next_scale()

# ============================================================================
# MGC入力
# ============================================================================

elif st.session_state.current_scale == 1:
    st.header("MG Composite (0-50点)")
    st.caption("各項目について、該当するスコアを選択してください")

    for item in MGC_ITEMS:
        key = f"mgc_{item['key']}"
        current_score = get_score(key)

        marker = " ⚡" if item['key'] in ['speech', 'chewing', 'swallowing', 'respiration'] else ""
        st.markdown(f'<div class="item-label">{item["name"]}{marker}<br><small style="color: #8e8e93;">{item.get("description", "")}</small></div>', unsafe_allow_html=True)

        # st.radioで横並び、実際の点数を表示
        selected = st.radio(
            label=item['name'],
            options=list(range(len(item['values']))),
            format_func=lambda x: str(item['values'][x]),
            index=current_score,
            horizontal=True,
            key=key,
            label_visibility="collapsed"
        )

        st.session_state.scores[key] = selected

    total = calculate_total(MGC_ITEMS, 'mgc')
    st.markdown(f'<div class="total-score" style="color: #34c759;">合計: {total}/50点</div>', unsafe_allow_html=True)
    st.caption("⚡印の項目はMG-ADLから自動反映されます")

    st.markdown("---")
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

elif st.session_state.current_scale == 2:
    st.header("MGQOL-15r (0-30点)")
    st.caption("各質問について、該当するスコアを選択してください")

    for item in MGQOL_ITEMS:
        key = f"mgqol_{item['key']}"
        current_score = get_score(key)

        st.markdown(f'<div class="item-label">{item["name"]}</div>', unsafe_allow_html=True)

        # st.radioで横並び（0-2点）
        selected = st.radio(
            label=item['name'],
            options=[0, 1, 2],
            index=current_score,
            horizontal=True,
            key=key,
            label_visibility="collapsed"
        )

        st.session_state.scores[key] = selected

    total = calculate_total(MGQOL_ITEMS, 'mgqol')
    st.markdown(f'<div class="total-score" style="color: #af52de;">合計: {total}/30点</div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← MGCへ戻る", use_container_width=True):
            prev_scale()
    with col3:
        if st.button("結果へ →", type="primary", use_container_width=True):
            next_scale()

# ============================================================================
# 予測結果タブ
# ============================================================================

elif st.session_state.current_scale == 3:
    st.header("予測結果")

    if st.session_state.prediction_results is None:
        with st.spinner("予測中..."):
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

            except FileNotFoundError as e:
                st.error(f"❌ ファイルエラー: {e}")
            except Exception as e:
                st.error(f"❌ 予測エラー: {e}")
                st.exception(e)

    if st.session_state.prediction_results is not None:
        results = st.session_state.prediction_results

        st.markdown("---")

        ensemble_prob = results['ensemble']
        classification = results['classification']
        total_votes = results['total_mm_votes']

        color = "#4CAF50" if classification == "MM or better" else "#F44336"

        st.markdown(f"""
        <div style='background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;'>
            <h2 style='color: white; margin: 0;'>予測結果</h2>
            <h1 style='color: white; margin: 10px 0; font-size: 48px;'>{classification}</h1>
            <h3 style='color: white; margin: 0;'>{total_votes}/3 モデルがMM or better</h3>
            <p style='color: white; margin: 10px 0; font-size: 14px;'>アンサンブル確率: {ensemble_prob:.1%}</p>
        </div>
        """, unsafe_allow_html=True)

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

        st.markdown("### 各モデルの予測")

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
                pred = vote_info['prediction']

                badge_color = "#4CAF50" if pred == "MM or better" else "#F44336"
                icon = "✓" if pred == "MM or better" else "✗"

                st.markdown(f"""
                <div style='padding: 15px; border: 2px solid {badge_color}; border-radius: 10px; text-align: center;'>
                    <div style='font-weight: bold; font-size: 14px; margin-bottom: 5px;'>{model_name}</div>
                    <div style='font-size: 24px; font-weight: bold; color: {badge_color};'>{prob:.1%}</div>
                    <div style='font-size: 20px; margin-top: 5px;'>{icon}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("### モジュールスコア")

        module_df = results['module_scores'].copy()
        module_df.columns = [col.replace('module ', '').title() for col in module_df.columns]

        cols = st.columns(4)
        for i, (col_name, value) in enumerate(module_df.iloc[0].items()):
            with cols[i]:
                st.metric(col_name, f"{value:.4f}")

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

        st.pyplot(fig, use_container_width=True)

    else:
        st.info("予測を実行中です...")

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← MGQOLへ戻る", use_container_width=True):
            prev_scale()

# ============================================================================
# リセットボタン（全画面共通）
# ============================================================================

st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🔄 全データリセット", use_container_width=True):
        reset_all_scores()

# ============================================================================
# フッター
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>MG予測システム v2.0 | 重症筋無力症 MM予測</small>
</div>
""", unsafe_allow_html=True)
