"""
Streamlit MG Prediction App
重症筋無力症（MG）予測システム - iOS風モバイルファーストUI
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
    layout="centered",  # centeredレイアウトでモバイルに最適化
    initial_sidebar_state="collapsed"
)

# カスタムCSS - iOS風デザイン、モバイルファースト
st.markdown("""
<style>
    /* リセットとベース設定 */
    * {
        box-sizing: border-box;
        -webkit-tap-highlight-color: transparent;
    }

    /* システムフォント */
    html, body {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Icons",
                     "Helvetica Neue", "Helvetica", "Arial", sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* 背景グラデーション */
    .stApp {
        background: linear-gradient(180deg, #F0F0F3 0%, #FFFFFF 100%);
        min-height: 100vh;
    }

    /* 横スクロール完全防止 */
    html, body, .stApp, .main {
        overflow-x: hidden !important;
        width: 100% !important;
    }

    /* メインコンテナ - モバイルファースト */
    .main .block-container {
        max-width: 600px !important;
        margin: 0 auto;
        padding: 1rem !important;
        width: 100% !important;
    }

    /* タイトル */
    h1 {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    h2 {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1c1c1e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    h3 {
        font-size: 1.2rem;
        font-weight: 600;
        color: #3c3c43;
    }

    /* 進捗インジケーター */
    .stColumns {
        margin-bottom: 1.5rem;
    }

    /* カード風の項目コンテナ */
    .item-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.04);
    }

    .item-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1c1c1e;
        margin-bottom: 12px;
        display: block;
    }

    .item-description {
        font-size: 0.9rem;
        color: #8e8e93;
        margin-top: 4px;
    }

    /* ラジオボタングループ - iOS風大型ボタン */
    div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        gap: 8px !important;
        width: 100% !important;
        margin: 12px 0 !important;
    }

    /* 各ラジオボタン */
    div[role="radiogroup"] > label {
        flex: 1 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* ラジオボタンをiOS風ボタンに */
    div[role="radiogroup"] > label > div {
        width: 100% !important;
        min-height: 60px !important;
        padding: 16px 8px !important;
        text-align: center !important;
        border: 2px solid #E5E5EA !important;
        border-radius: 12px !important;
        background: #FFFFFF !important;
        color: #1c1c1e !important;
        font-weight: 600 !important;
        font-size: 1.25rem !important;
        cursor: pointer !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        user-select: none !important;
        -webkit-user-select: none !important;
        position: relative !important;
        overflow: hidden !important;
    }

    /* タップ時のリップルエフェクト */
    div[role="radiogroup"] > label > div:active {
        transform: scale(0.96) !important;
    }

    /* 選択状態 - 鮮やかな青 */
    div[role="radiogroup"] > label > div[data-checked="true"] {
        background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%) !important;
        border-color: transparent !important;
        color: white !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 12px rgba(0, 122, 255, 0.3) !important;
    }

    /* ラジオボタンの丸を非表示 */
    div[role="radiogroup"] input[type="radio"] {
        display: none !important;
    }

    /* ホバーエフェクト（デスクトップ） */
    @media (hover: hover) {
        div[role="radiogroup"] > label > div:hover {
            background: #F2F2F7 !important;
            border-color: #007AFF !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        }

        div[role="radiogroup"] > label > div[data-checked="true"]:hover {
            background: linear-gradient(135deg, #0051D5 0%, #4A49C5 100%) !important;
            box-shadow: 0 6px 16px rgba(0, 122, 255, 0.4) !important;
        }
    }

    /* 合計スコア表示 */
    .total-score {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        padding: 20px;
        border-radius: 20px;
        margin: 24px 0;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
    }

    .total-score.mgc {
        background: linear-gradient(135deg, #34C759 0%, #30D158 100%);
        box-shadow: 0 8px 24px rgba(52, 199, 89, 0.3);
    }

    .total-score.mgqol {
        background: linear-gradient(135deg, #AF52DE 0%, #BF5AF2 100%);
        box-shadow: 0 8px 24px rgba(175, 82, 222, 0.3);
    }

    /* ナビゲーションボタン */
    div.stButton > button {
        width: 100% !important;
        min-height: 56px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        border-radius: 14px !important;
        border: none !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
    }

    div.stButton > button:active {
        transform: scale(0.98) !important;
    }

    /* プライマリボタン */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(0, 122, 255, 0.3) !important;
    }

    div.stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 16px rgba(0, 122, 255, 0.4) !important;
    }

    /* セカンダリボタン */
    div.stButton > button[kind="secondary"] {
        background: #F2F2F7 !important;
        color: #007AFF !important;
    }

    /* タブレット対応 */
    @media screen and (min-width: 600px) {
        div[role="radiogroup"] {
            gap: 10px !important;
        }

        div[role="radiogroup"] > label > div {
            min-height: 64px !important;
            font-size: 1.3rem !important;
        }

        .item-card {
            padding: 20px;
            margin-bottom: 20px;
        }
    }

    /* スマホ対応（標準） */
    @media screen and (max-width: 599px) {
        .main .block-container {
            padding: 0.75rem !important;
        }

        h1 {
            font-size: 1.75rem;
        }

        h2 {
            font-size: 1.3rem;
        }

        .item-card {
            padding: 14px;
            margin-bottom: 14px;
        }

        .item-label {
            font-size: 1rem;
        }

        div[role="radiogroup"] {
            gap: 6px !important;
        }

        div[role="radiogroup"] > label > div {
            min-height: 56px !important;
            font-size: 1.15rem !important;
            padding: 14px 6px !important;
            border-radius: 10px !important;
        }

        .total-score {
            font-size: 1.3rem;
            padding: 18px;
        }
    }

    /* 小型スマホ対応 */
    @media screen and (max-width: 400px) {
        .main .block-container {
            padding: 0.5rem !important;
        }

        h1 {
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }

        .item-card {
            padding: 12px;
            margin-bottom: 12px;
            border-radius: 12px;
        }

        div[role="radiogroup"] {
            gap: 4px !important;
        }

        div[role="radiogroup"] > label > div {
            min-height: 52px !important;
            font-size: 1.1rem !important;
            padding: 12px 4px !important;
            border-width: 1.5px !important;
        }

        .total-score {
            font-size: 1.2rem;
            padding: 16px;
            border-radius: 16px;
        }

        div.stButton > button {
            min-height: 52px !important;
            font-size: 1rem !important;
        }
    }

    /* 極小デバイス対応 (iPhone SE等) */
    @media screen and (max-width: 350px) {
        div[role="radiogroup"] {
            gap: 3px !important;
        }

        div[role="radiogroup"] > label > div {
            min-height: 48px !important;
            font-size: 1rem !important;
            padding: 10px 2px !important;
            border-radius: 8px !important;
        }
    }

    /* スクロールバー美化 */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }

    ::-webkit-scrollbar-track {
        background: #F2F2F7;
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb {
        background: #C7C7CC;
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #8E8E93;
    }

    /* アニメーション */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .item-card {
        animation: slideIn 0.3s ease-out;
    }

    /* エラー・成功メッセージ */
    .stAlert {
        border-radius: 12px;
        padding: 12px 16px;
    }

    /* メトリクス */
    [data-testid="metric-container"] {
        background: white;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
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

st.title("MG予測システム")

# 進捗インジケーター
progress_labels = ["MG-ADL", "MGC", "MGQOL", "結果"]
progress_emojis = ["📝", "📊", "💭", "✨"]
cols = st.columns(4)
for i, (label, emoji) in enumerate(zip(progress_labels, progress_emojis)):
    with cols[i]:
        if i == st.session_state.current_scale:
            st.markdown(f"**{emoji}**<br>**{label}**", unsafe_allow_html=True)
        elif i < st.session_state.current_scale:
            st.markdown(f"✅<br>{label}", unsafe_allow_html=True)
        else:
            st.markdown(f"⭕<br>{label}", unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# MG-ADL入力
# ============================================================================

if st.session_state.current_scale == 0:
    st.header("📝 MG-ADL")
    st.caption("日常生活動作の評価（0-24点）")

    for item in MGADL_ITEMS:
        key = f"adl_{item['key']}"
        current_score = get_score(key)

        # カード風デザイン
        marker = " ⚡" if item['key'] in ['speech', 'chewing', 'swallowing', 'respiration'] else ""
        st.markdown(f'<div class="item-card">', unsafe_allow_html=True)
        st.markdown(f'<span class="item-label">{item["name"]}{marker}</span>', unsafe_allow_html=True)

        # st.radioで横並びボタン
        selected = st.radio(
            label=item['name'],
            options=list(range(len(item['options']))),
            index=current_score,
            horizontal=True,
            key=key,
            label_visibility="collapsed"
        )

        st.session_state.scores[key] = selected

        # 自動同期
        if item['key'] in ['speech', 'chewing', 'swallowing', 'respiration']:
            sync_adl_to_mgc()

        st.markdown('</div>', unsafe_allow_html=True)

    # 合計点
    total = calculate_total(MGADL_ITEMS, 'adl')
    st.markdown(f'<div class="total-score">合計: {total}/24点</div>', unsafe_allow_html=True)

    if st.session_state.scores:
        st.caption("⚡ 自動的にMGCに反映されます")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col2:
        if st.button("次へ進む →", type="primary", use_container_width=True):
            next_scale()

# ============================================================================
# MGC入力
# ============================================================================

elif st.session_state.current_scale == 1:
    st.header("📊 MG Composite")
    st.caption("総合評価スケール（0-50点）")

    for item in MGC_ITEMS:
        key = f"mgc_{item['key']}"
        current_score = get_score(key)

        marker = " ⚡" if item['key'] in ['speech', 'chewing', 'swallowing', 'respiration'] else ""

        st.markdown(f'<div class="item-card">', unsafe_allow_html=True)
        st.markdown(f'<span class="item-label">{item["name"]}{marker}</span>', unsafe_allow_html=True)
        if item.get("description"):
            st.markdown(f'<span class="item-description">{item["description"]}</span>', unsafe_allow_html=True)

        # 実際の点数を表示
        selected = st.radio(
            label=item['name'],
            options=list(range(len(item['values']))),
            format_func=lambda x, values=item['values']: str(values[x]),
            index=current_score,
            horizontal=True,
            key=key,
            label_visibility="collapsed"
        )

        st.session_state.scores[key] = selected
        st.markdown('</div>', unsafe_allow_html=True)

    total = calculate_total(MGC_ITEMS, 'mgc')
    st.markdown(f'<div class="total-score mgc">合計: {total}/50点</div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← 戻る", type="secondary", use_container_width=True):
            prev_scale()
    with col2:
        if st.button("次へ進む →", type="primary", use_container_width=True):
            next_scale()

# ============================================================================
# MGQOL入力
# ============================================================================

elif st.session_state.current_scale == 2:
    st.header("💭 MGQOL-15r")
    st.caption("生活の質評価（0-30点）")

    for item in MGQOL_ITEMS:
        key = f"mgqol_{item['key']}"
        current_score = get_score(key)

        st.markdown(f'<div class="item-card">', unsafe_allow_html=True)
        st.markdown(f'<span class="item-label">{item["name"]}</span>', unsafe_allow_html=True)

        # 0-2点の選択
        options_labels = ["0", "1", "2"]
        selected = st.radio(
            label=item['name'],
            options=[0, 1, 2],
            format_func=lambda x: options_labels[x],
            index=current_score,
            horizontal=True,
            key=key,
            label_visibility="collapsed"
        )

        st.session_state.scores[key] = selected
        st.markdown('</div>', unsafe_allow_html=True)

    total = calculate_total(MGQOL_ITEMS, 'mgqol')
    st.markdown(f'<div class="total-score mgqol">合計: {total}/30点</div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← 戻る", type="secondary", use_container_width=True):
            prev_scale()
    with col2:
        if st.button("予測を実行 →", type="primary", use_container_width=True):
            next_scale()

# ============================================================================
# 予測結果
# ============================================================================

elif st.session_state.current_scale == 3:
    st.header("✨ 予測結果")

    if st.session_state.prediction_results is None:
        with st.spinner("AIが解析中..."):
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

        ensemble_prob = results['ensemble']
        classification = results['classification']
        total_votes = results['total_mm_votes']

        # 結果カード
        if classification == "MM or better":
            result_color = "linear-gradient(135deg, #34C759 0%, #30D158 100%)"
            result_emoji = "✅"
        else:
            result_color = "linear-gradient(135deg, #FF3B30 0%, #FF453A 100%)"
            result_emoji = "⚠️"

        st.markdown(f"""
        <div style='
            background: {result_color};
            padding: 24px;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
            margin-bottom: 24px;
        '>
            <div style='font-size: 48px; margin-bottom: 12px;'>{result_emoji}</div>
            <h2 style='color: white; margin: 0; font-size: 28px;'>{classification}</h2>
            <p style='color: white; margin: 12px 0 0 0; font-size: 16px; opacity: 0.95;'>
                {total_votes}/3 モデルが予測<br>
                確率: {ensemble_prob:.1%}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # 入力スコアサマリー
        st.markdown("### 📊 入力スコア")
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

        # モデル詳細
        st.markdown("### 🤖 AI解析詳細")

        cols = st.columns(3)
        for i, (model_name, vote_info) in enumerate(results['model_votes'].items()):
            with cols[i]:
                prob = vote_info['probability']
                pred = vote_info['prediction']

                if pred == "MM or better":
                    card_bg = "#E8F5E9"
                    text_color = "#2E7D32"
                    icon = "✓"
                else:
                    card_bg = "#FFEBEE"
                    text_color = "#C62828"
                    icon = "✗"

                st.markdown(f"""
                <div style='
                    background: {card_bg};
                    padding: 16px;
                    border-radius: 12px;
                    text-align: center;
                '>
                    <div style='font-size: 24px; color: {text_color};'>{icon}</div>
                    <div style='font-weight: 600; color: #666; font-size: 12px; margin: 8px 0 4px 0;'>
                        {model_name}
                    </div>
                    <div style='font-size: 20px; font-weight: 700; color: {text_color};'>
                        {prob:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # モジュールスコア
        st.markdown("### 🎯 モジュール解析")
        module_df = results['module_scores'].copy()
        module_df.columns = [col.replace('module ', '').title() for col in module_df.columns]

        cols = st.columns(2)
        for i, (col_name, value) in enumerate(module_df.iloc[0].items()):
            with cols[i % 2]:
                st.metric(col_name, f"{value:.4f}")

        # レーダーチャート
        st.markdown("### 📈 比較チャート")
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

        # 詳細情報
        with st.expander("🔍 詳細データ"):
            st.write("**モデル別予測値:**")
            for model_name, vote_info in results['model_votes'].items():
                st.write(f"- {model_name}: {vote_info['probability']:.4f} "
                        f"(カットオフ: {vote_info['cutoff']:.4f})")

            st.write("\n**5-fold交差検証:**")
            for model_name, probs in results['predictions'].items():
                st.write(f"- {model_name}: {[f'{p:.3f}' for p in probs]}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← 入力に戻る", type="secondary", use_container_width=True):
            prev_scale()

# ============================================================================
# リセットボタン
# ============================================================================

st.markdown("---")
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    if st.button("🔄 リセット", use_container_width=True):
        reset_all_scores()

# ============================================================================
# フッター
# ============================================================================

st.markdown("""
<div style='text-align: center; margin-top: 40px; padding: 20px; color: #8E8E93; font-size: 12px;'>
    MG予測システム v3.0<br>
    Myasthenia Gravis Prediction
</div>
""", unsafe_allow_html=True)