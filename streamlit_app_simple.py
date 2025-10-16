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

# カスタムCSS - iOS風デザイン
st.markdown("""
<style>
    /* iOS風デザイン */
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;500;600;700&display=swap');

    body {
        font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
        background-color: #f2f2f7;
    }

    /* 全体のコンテナ */
    .main .block-container {
        max-width: 800px;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    /* ヘッダー */
    h1, h2, h3 {
        font-weight: 600;
        color: #1c1c1e;
    }

    /* 進捗インジケーター */
    .stMarkdown {
        margin-bottom: 0.5rem;
    }

    /* スコアボタングループのスタイル */
    div.stButton {
        margin: 0;
        padding: 0;
    }

    div.stButton > button {
        width: 100%;
        height: 60px;
        font-size: 1.2em;
        font-weight: 600;
        border-radius: 10px;
        border: 2px solid #e5e5ea;
        background-color: white;
        color: #1c1c1e;
        transition: all 0.2s ease;
        margin: 2px;
    }

    div.stButton > button:hover {
        background-color: #f2f2f7;
        border-color: #007aff;
    }

    /* 選択されたボタン（data-baseweb属性を使って判定） */
    div.stButton > button[kind="secondary"] {
        background-color: white;
        border-color: #e5e5ea;
        color: #1c1c1e;
    }

    /* プライマリボタン（選択済み） */
    div.stButton > button[kind="primary"] {
        background-color: #007aff;
        border-color: #007aff;
        color: white;
        font-weight: 700;
    }

    /* MGC用の緑色 */
    .mgc-selected button[kind="primary"] {
        background-color: #34c759 !important;
        border-color: #34c759 !important;
    }

    /* MGQOL用の紫色 */
    .mgqol-selected button[kind="primary"] {
        background-color: #af52de !important;
        border-color: #af52de !important;
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
    div.stButton > button[data-testid="baseButton-primary"] {
        min-height: 60px;
        font-size: 1.3em;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
    }

    /* モバイル対応 */
    @media screen and (max-width: 768px) {
        .main .block-container {
            padding: 0.5rem;
        }

        div.stButton > button {
            height: 55px;
            font-size: 1.1em;
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

def set_score(key, value):
    """スコアを設定"""
    st.session_state.scores[key] = value

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
    st.session_state.current_scale = 0  # 0: ADL, 1: MGC, 2: MGQOL, 3: Results

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

def go_to_scale(scale_index):
    """指定のスケールに移動"""
    st.session_state.current_scale = scale_index
    st.rerun()

# ============================================================================
# メインUI
# ============================================================================

st.title("MG予測システム")

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
    st.caption("各項目について、該当するスコアをタップしてください")

    for item in MGADL_ITEMS:
        key = f"adl_{item['key']}"
        current_score = get_score(key)
        max_val = len(item['options']) - 1

        # 自動反映される項目には印をつける
        marker = " ⚡" if item['key'] in ['speech', 'chewing', 'swallowing', 'respiration'] else ""
        st.markdown(f'<div class="item-label">{item["name"]}{marker}</div>', unsafe_allow_html=True)

        # スコアボタンを横並びで表示
        cols = st.columns(max_val + 1)
        for i in range(max_val + 1):
            with cols[i]:
                button_type = "primary" if current_score == i else "secondary"
                if st.button(f"{i}", key=f"btn_{key}_{i}", type=button_type, use_container_width=True):
                    set_score(key, i)
                    # 自動的にMGCに反映
                    sync_adl_to_mgc()
                    st.rerun()

    # 合計点表示
    total = calculate_total(MGADL_ITEMS, 'adl')
    st.markdown(f'<div class="total-score">合計: {total}/24点</div>', unsafe_allow_html=True)
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

elif st.session_state.current_scale == 1:
    st.header("MG Composite (0-50点)")
    st.caption("各項目について、該当するスコアをタップしてください")

    for item in MGC_ITEMS:
        key = f"mgc_{item['key']}"
        current_score = get_score(key)
        max_val = len(item['values']) - 1

        marker = " ⚡" if item['key'] in ['speech', 'chewing', 'swallowing', 'respiration'] else ""
        st.markdown(f'<div class="item-label">{item["name"]}{marker}<br><small style="color: #8e8e93;">{item.get("description", "")}</small></div>', unsafe_allow_html=True)

        # スコアボタンを横並びで表示（実際の点数を表示）
        cols = st.columns(max_val + 1)
        for i in range(max_val + 1):
            with cols[i]:
                actual_score = item['values'][i]
                button_type = "primary" if current_score == i else "secondary"
                if st.button(f"{actual_score}", key=f"btn_{key}_{i}", type=button_type, use_container_width=True):
                    set_score(key, i)
                    st.rerun()

    # 合計点表示
    total = calculate_total(MGC_ITEMS, 'mgc')
    st.markdown(f'<div class="total-score" style="color: #34c759;">合計: {total}/50点</div>', unsafe_allow_html=True)
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

elif st.session_state.current_scale == 2:
    st.header("MGQOL-15r (0-30点)")
    st.caption("各質問について、該当するスコアをタップしてください")

    for item in MGQOL_ITEMS:
        key = f"mgqol_{item['key']}"
        current_score = get_score(key)

        st.markdown(f'<div class="item-label">{item["name"]}</div>', unsafe_allow_html=True)

        # スコアボタンを横並びで表示（0-2点）
        cols = st.columns(3)
        score_labels = ["0: 全くそうは思わない", "1: 少しそう思う", "2: 強くそう思う"]
        for i in range(3):
            with cols[i]:
                button_type = "primary" if current_score == i else "secondary"
                if st.button(f"{i}", key=f"btn_{key}_{i}", type=button_type, use_container_width=True):
                    set_score(key, i)
                    st.rerun()

    # 合計点表示
    total = calculate_total(MGQOL_ITEMS, 'mgqol')
    st.markdown(f'<div class="total-score" style="color: #af52de;">合計: {total}/30点</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ナビゲーションボタン
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

    # 予測を自動実行（まだ実行されていない場合）
    if st.session_state.prediction_results is None:
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
                st.rerun()

            except FileNotFoundError as e:
                st.error(f"❌ ファイルエラー: {e}")
            except Exception as e:
                st.error(f"❌ 予測エラー: {e}")
                st.exception(e)

    # 結果表示
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

        # matplotlib チャートを表示
        st.pyplot(fig, use_container_width=True)

    else:
        st.info("予測を実行中です...")

    # ナビゲーションボタン
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
    reset_btn = st.button("🔄 全データリセット", use_container_width=True)

if reset_btn:
    reset_all_scores()

# ============================================================================
# フッター
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>MG予測システム v1.0 | 重症筋無力症 MM予測</small>
</div>
""", unsafe_allow_html=True)
