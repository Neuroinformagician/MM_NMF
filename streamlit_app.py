"""
Streamlit MG Prediction App
重症筋無力症（MG）予測システム
"""

import streamlit as st
import pandas as pd
import numpy as np

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

with st.sidebar:
    st.header("📋 患者データ入力")

    # MG-ADL入力
    with st.expander("🔵 MG-ADL (8項目)", expanded=True):
        for item in MGADL_ITEMS:
            key = f"adl_{item['key']}"
            default_value = st.session_state.scores.get(key, 0)

            st.session_state.scores[key] = st.selectbox(
                item['name'],
                options=[0, 1, 2, 3],
                format_func=lambda x, opts=item['options']: opts[x],
                index=default_value,
                key=f"input_{key}"
            )

        adl_total = calculate_adl_total()
        st.metric("合計点", f"{adl_total}/24点")

    # MG Composite入力
    with st.expander("🟢 MG Composite (10項目)", expanded=False):
        # ADLから自動反映ボタン
        if st.button("🔄 ADLから重複項目を反映", key="sync_adl"):
            sync_adl_to_mgc()
            st.rerun()

        for item in MGC_ITEMS:
            key = f"mgc_{item['key']}"
            default_value = st.session_state.scores.get(key, 0)

            # ADLから反映される項目は表示を変える
            if 'from_adl' in item:
                label = f"{item['name']} ⚡"
                help_text = f"ADLの「{item['from_adl']}」から自動反映可能"
            else:
                label = item['name']
                help_text = item['description']

            st.session_state.scores[key] = st.selectbox(
                label,
                options=[0, 1, 2, 3],
                format_func=lambda x, opts=item['options']: opts[x],
                index=default_value,
                key=f"input_{key}",
                help=help_text
            )

        mgc_total = calculate_mgc_total()
        st.metric("合計点", f"{mgc_total}/50点")

    # MGQOL-15r入力
    with st.expander("🟣 MGQOL-15r (15項目)", expanded=False):
        for item in MGQOL_ITEMS:
            key = f"mgqol_{item['key']}"
            default_value = st.session_state.scores.get(key, 0)

            st.session_state.scores[key] = st.selectbox(
                f"Q{item['key'][1:]}. {item['name'][:20]}...",
                options=[0, 1, 2],
                format_func=lambda x: MGQOL_OPTIONS[x],
                index=default_value,
                key=f"input_{key}",
                help=item['name']
            )

        mgqol_total = calculate_mgqol_total()
        st.metric("合計点", f"{mgqol_total}/30点")

    # 予測・リセットボタン
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        predict_btn = st.button("🔮 予測実行", type="primary", use_container_width=True)

    with col2:
        reset_btn = st.button("🔄 リセット", use_container_width=True)

    if reset_btn:
        reset_all_scores()

# ============================================================================
# メインエリア
# ============================================================================

st.title("🏥 MG予測システム")
st.markdown("重症筋無力症（MG）患者の**MM（Minimal Manifestations）予測**システム")

# 入力サマリー
st.subheader("📊 入力サマリー")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🔵 MG-ADL", f"{adl_total}/24点")

with col2:
    st.metric("🟢 MG Composite", f"{mgc_total}/50点")

with col3:
    st.metric("🟣 MGQOL-15r", f"{mgqol_total}/30点")

st.markdown("---")

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

    # モジュールスコア表示
    st.subheader("🎯 予測結果")

    st.markdown("#### モジュールスコア")
    module_df = results['module_scores'].copy()
    module_df.columns = [col.replace('module ', '').title() for col in module_df.columns]

    # スタイリング
    styled_df = module_df.style.format("{:.4f}").background_gradient(
        cmap='YlOrRd', axis=1
    )
    st.dataframe(styled_df, use_container_width=True)

    # 各モデルの予測確率
    st.markdown("#### 各モデル予測確率")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("SVM", f"{results['mean_predictions']['SVM']:.1%}")

    with col2:
        st.metric("Logistic Regression", f"{results['mean_predictions']['Logistic Regression']:.1%}")

    with col3:
        st.metric("Random Forest", f"{results['mean_predictions']['Random Forest']:.1%}")

    with col4:
        st.metric("Naive Bayes", f"{results['mean_predictions']['Naive Bayes']:.1%}")

    # アンサンブル結果（強調表示）
    st.markdown("---")
    st.markdown("### 📊 アンサンブル予測")

    ensemble_prob = results['ensemble']
    classification = results['classification']

    col1, col2 = st.columns([1, 2])

    with col1:
        # 大きく表示
        if ensemble_prob > 0.5:
            st.success(f"## {ensemble_prob:.1%}")
            st.success(f"### {classification}")
        else:
            st.warning(f"## {ensemble_prob:.1%}")
            st.warning(f"### {classification}")

    with col2:
        # 説明
        if ensemble_prob > 0.5:
            st.info("""
            **MM or better (寛解状態)** と予測されました。
            - アンサンブル確率が50%を超えています
            - 現在の症状は軽症〜中等症と考えられます
            """)
        else:
            st.warning("""
            **non MM (非寛解状態)** と予測されました。
            - アンサンブル確率が50%未満です
            - 症状のコントロールに注意が必要です
            """)

    # レーダーチャート
    st.markdown("---")
    st.subheader("📈 モジュールスコア比較（レーダーチャート）")

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

    # 比較テーブル
    st.markdown("---")
    st.subheader("📋 詳細比較")

    comparison_table = create_module_comparison_table(
        patient_scores=patient_scores,
        mm_avg=comparison['mm_avg'],
        non_mm_avg=comparison['non_mm_avg'],
        module_names=comparison['module_names']
    )

    st.dataframe(comparison_table, use_container_width=True)

    # モジュール別評価
    st.markdown("#### モジュール別評価")

    assessments = create_module_assessment_text(
        patient_scores=patient_scores,
        mm_avg=comparison['mm_avg'],
        non_mm_avg=comparison['non_mm_avg'],
        module_names=comparison['module_names']
    )

    for assessment in assessments:
        color = assessment['color']
        if color == 'green':
            st.success(f"{assessment['icon']} **{assessment['module']}**: {assessment['score']:.4f} — {assessment['status']}")
        elif color == 'red':
            st.error(f"{assessment['icon']} **{assessment['module']}**: {assessment['score']:.4f} — {assessment['status']}")
        else:
            st.info(f"{assessment['icon']} **{assessment['module']}**: {assessment['score']:.4f} — {assessment['status']}")

else:
    # 予測前の表示
    st.info("👈 左のサイドバーから患者データを入力し、「🔮 予測実行」ボタンをクリックしてください。")

    # サンプルデータボタン
    if st.button("📝 サンプルデータを読み込む"):
        # 症例予測.ipynbのサンプルデータ
        sample_data = {
            # MGC
            "mgc_ptosis": 0,
            "mgc_diplopia": 0,
            "mgc_eyelid_closure": 0,
            "mgc_speech": 0,
            "mgc_chewing": 0,
            "mgc_swallowing": 0,
            "mgc_respiration": 1,
            "mgc_neck": 1,
            "mgc_upper_limb": 0,
            "mgc_lower_limb": 0,
            # MGADL
            "adl_speech": 0,
            "adl_chewing": 0,
            "adl_swallowing": 0,
            "adl_respiration": 0,
            "adl_toothbrushing": 1,
            "adl_getting_up": 0,
            "adl_diplopia": 0,
            "adl_ptosis": 0,
            # MGQOL15r
            "mgqol_q1": 0,
            "mgqol_q2": 0,
            "mgqol_q3": 0,
            "mgqol_q4": 0,
            "mgqol_q5": 0,
            "mgqol_q6": 0,
            "mgqol_q7": 0,
            "mgqol_q8": 0,
            "mgqol_q9": 0,
            "mgqol_q10": 0,
            "mgqol_q11": 0,
            "mgqol_q12": 0,
            "mgqol_q13": 1,
            "mgqol_q14": 1,
            "mgqol_q15": 1
        }
        st.session_state.scores = sample_data
        st.rerun()

# ============================================================================
# フッター
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>MG予測システム v1.0 | 重症筋無力症 MM予測</small>
</div>
""", unsafe_allow_html=True)
