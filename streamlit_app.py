"""
Streamlit MG Prediction App
重症筋無力症（MG）予測システム
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from pathlib import Path

from streamlit_config import (
    MGADL_ITEMS, MGC_ITEMS, MGQOL_ITEMS, MGQOL_OPTIONS,
    check_required_files
)
from streamlit_predictor import MGPredictor
from streamlit_visualizer import (
    create_radar_chart,
    create_module_comparison_table,
    create_module_assessment_text
)

# ============================================================================
# ページ設定
# ============================================================================

st.set_page_config(
    page_title="MG予測システム",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def calculate_adl_total():
    """MG-ADL合計点を計算"""
    total = 0
    for item in MGADL_ITEMS:
        key = f"adl_{item['key']}"
        if key in st.session_state.scores:
            total += st.session_state.scores[key]
    return total


def calculate_mgc_total():
    """MG Composite合計点を計算"""
    total = 0
    for item in MGC_ITEMS:
        key = f"mgc_{item['key']}"
        if key in st.session_state.scores:
            index = st.session_state.scores[key]
            total += item['values'][index]
    return total


def calculate_mgqol_total():
    """MGQOL-15r合計点を計算"""
    total = 0
    for item in MGQOL_ITEMS:
        key = f"mgqol_{item['key']}"
        if key in st.session_state.scores:
            total += st.session_state.scores[key]
    return total


def reset_all_scores():
    """全スコアをリセット"""
    st.session_state.scores = {}
    st.session_state.prediction_results = None
    st.rerun()


def sync_adl_to_mgc():
    """ADLからMGCへの自動同期"""
    adl_to_mgc_map = {
        "speech": "speech",
        "chewing": "chewing",
        "swallowing": "swallowing",
        "respiration": "respiration"
    }

    for adl_key, mgc_key in adl_to_mgc_map.items():
        adl_full_key = f"adl_{adl_key}"
        mgc_full_key = f"mgc_{mgc_key}"

        if adl_full_key in st.session_state.scores:
            st.session_state.scores[mgc_full_key] = st.session_state.scores[adl_full_key]


# ============================================================================
# サイドバー: 入力UI
# ============================================================================

# タップ入力UI（サイドバーなし、メイン表示）
st.header("患者データ入力")

# HTMLコンポーネントを読み込み
html_path = Path(__file__).parent / "streamlit_tap_input.html"
with open(html_path, 'r', encoding='utf-8') as f:
    html_content = f.read()

# タップ入力コンポーネント
tap_scores = components.html(
    html_content,
    height=650,
    scrolling=False
)

# デバッグ: HTMLコンポーネントからの戻り値を表示
st.write("### デバッグ情報")
st.write("HTMLコンポーネントからの戻り値:")
st.write(f"型: {type(tap_scores)}")
st.write(f"値: {tap_scores}")

# タップ入力から値を取得
if tap_scores is not None and isinstance(tap_scores, dict):
    st.write("✅ 辞書型のデータを受信しました")
    # 値が変更された場合のみ更新
    if tap_scores != st.session_state.scores:
        st.session_state.scores = tap_scores.copy()
        st.write("✅ セッション状態を更新しました")

        # ADLからMGCへの自動同期
        sync_adl_to_mgc()
        st.write("✅ ADL→MGC同期を実行しました")
else:
    st.write("❌ HTMLコンポーネントからデータを受信していません")

# デバッグ表示（現在のスコア）
st.write("### 現在のセッション状態")
st.write(f"session_state.scores: {st.session_state.scores}")
if st.session_state.scores:
    with st.expander("詳細データ確認"):
        st.write(st.session_state.scores)
        st.write(f"MG-ADL合計: {calculate_adl_total()}")
        st.write(f"MGC合計: {calculate_mgc_total()}")
        st.write(f"MGQOL合計: {calculate_mgqol_total()}")

st.markdown("---")

# 予測・リセットボタン
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    predict_btn = st.button("予測実行", type="primary", use_container_width=True)

with col2:
    reset_btn = st.button("リセット", use_container_width=True)

with col3:
    sync_btn = st.button("ADL→MGC同期", use_container_width=True)

if reset_btn:
    reset_all_scores()

if sync_btn:
    sync_adl_to_mgc()
    st.rerun()

# ============================================================================
# メインエリア
# ============================================================================

st.title("MG予測システム")

# ============================================================================
# 予測実行
# ============================================================================

if predict_btn:
    with st.spinner("予測中..."):
        try:
            # 必須ファイルチェック
            check_required_files()

            # Predictorの初期化
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

    # アンサンブル予測（大きく表示）
    ensemble_prob = results['ensemble']
    classification = results['classification']

    # 予測結果表示
    st.markdown(f"### 予測結果")
    st.markdown(f"**MM or betterの確率:** {ensemble_prob:.1%}")
    st.markdown(f"**判定:** {classification}")

    st.markdown("---")

    # モジュールスコア表示（シンプルに）
    st.markdown("### モジュールスコア")

    module_df = results['module_scores'].copy()
    module_df.columns = [col.replace('module ', '').title() for col in module_df.columns]

    # 4つのメトリクスで表示
    cols = st.columns(4)
    for i, (col_name, value) in enumerate(module_df.iloc[0].items()):
        with cols[i]:
            st.metric(col_name, f"{value:.4f}")

    st.markdown("---")

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
    # 予測前の表示
    st.info("左のサイドバーから患者データを入力し、「予測実行」ボタンをクリックしてください。")

# ============================================================================
# フッター
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>MG予測システム v1.0 | 重症筋無力症 MM予測</small>
</div>
""", unsafe_allow_html=True)
