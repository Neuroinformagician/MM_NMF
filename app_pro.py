"""
MG MM Predictor — Pro (clinical light UI)

判定ロジックは app_simple.py と完全に同一（数値・アンサンブル手法とも）。
デザインのみを、落ち着いた臨床ソフト調に整えたスクリーンショット用アプリ。

起動:
    streamlit run app_pro.py
"""

import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="MG MM Predictor",
    page_icon=":material/monitor_heart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =====================================================================================
# モデル定義（app_simple.py と同一）
# =====================================================================================
MODELS = {
    "svm": ("SVM", "SVM"),
    "logistic_regression": ("ロジスティック回帰", "Logistic Regression"),
    "random_forest": ("ランダムフォレスト", "Random Forest"),
    "naive_bayes": ("ナイーブベイズ", "GaussianNB"),
}
MODEL_EN = {
    "SVM": "SVM",
    "ロジスティック回帰": "Logistic Regression",
    "ランダムフォレスト": "Random Forest",
    "ナイーブベイズ": "Naive Bayes",
}
MODULES = ["module QOL", "module Diplopia", "module Ptosis", "module Systemic"]
MODULE_LABELS_JA = {
    "module QOL": "QOL",
    "module Diplopia": "複視",
    "module Ptosis": "眼瞼下垂",
    "module Systemic": "全身",
}

ITEM_LABELS = {
    "MGC": {
        "MGC_ptosis": "眼瞼下垂", "MGC_diplopia": "複視", "MGC_eyelid_closure": "眼瞼閉鎖",
        "MGC_speech": "構音障害", "MGC_chewing": "咀嚼", "MGC_swallowing": "嚥下",
        "MGC_respiration": "呼吸", "MGC_neck": "頸部", "MGC_upper_limb": "上肢",
        "MGC_lower_limb": "下肢",
    },
    "MGADL": {
        "MGADL_speech": "会話", "MGADL_chewing": "咀嚼", "MGADL_swallowing": "嚥下",
        "MGADL_respiration": "呼吸", "MGADL_toothbrushing": "歯磨き・整髪",
        "MGADL_getting_up": "立ち上がり", "MGADL_diplopia": "複視", "MGADL_ptosis": "眼瞼下垂",
    },
    "MGQOL15": {
        "MGQOL1_dissatisfaction": "不満足感", "MGQOL2_seeing": "視覚", "MGQOL3_eating": "食事",
        "MGQOL4_social_activity_restriction": "社会活動制限", "MGQOL5_hobby_entertainment": "趣味・娯楽",
        "MGQOL6_family_role": "家庭内役割", "MGQOL7_behavior_modification": "行動変更",
        "MGQOL8_work_impact": "仕事への影響", "MGQOL9_speaking": "会話", "MGQOL10_driving": "運転",
        "MGQOL11_feeling_down": "気分の落ち込み", "MGQOL12_walking": "歩行",
        "MGQOL13_quick_action": "素早い動作", "MGQOL14_mental_crushing": "精神的圧迫感",
        "MGQOL15_dressing": "着替え",
    },
}

GROUP_LABEL_EN = {"MGC": "MG Composite", "MGADL": "MG-ADL", "MGQOL15": "MG-QOL15"}


# =====================================================================================
# アーティファクト読み込み（app_simple.py と同一パイプライン）
# =====================================================================================
@st.cache_resource
def load_artifacts():
    with open("./out/feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    with open("./out/minmax_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("./out/nmf_model.pkl", "rb") as f:
        nmf = pickle.load(f)
    with open("./out/optimal_cutoffs.pkl", "rb") as f:
        cutoffs = pickle.load(f)
    with open("./out/group_statistics.pkl", "rb") as f:
        group_stats = pickle.load(f)

    folds = {}
    for prefix in MODELS:
        folds[prefix] = [
            pickle.load(open(f"./out/{prefix}_fold_{i}.pkl", "rb")) for i in range(5)
        ]

    item_max = {col: int(round(m)) for col, m in zip(feature_cols, scaler.data_max_)}
    return feature_cols, scaler, nmf, cutoffs, folds, item_max, group_stats


def predict(scores, feature_cols, scaler, nmf, cutoffs, folds):
    x = pd.DataFrame([[scores[c] for c in feature_cols]], columns=feature_cols)
    x_scaled = scaler.transform(x)
    W = nmf.transform(x_scaled)
    W_df = pd.DataFrame(W, columns=MODULES)

    results = {}
    for prefix, (disp, cut_key) in MODELS.items():
        prob = float(np.mean([m.predict_proba(W_df)[0, 1] for m in folds[prefix]]))
        results[disp] = {"prob": prob, "label": prob >= cutoffs[cut_key]}
    return W[0], results


# =====================================================================================
# プリセット（実際にパイプラインで確認済み: mild -> MM 92.5%, severe -> non-MM 4.7%）
# =====================================================================================
def build_preset_mild(feature_cols):
    scores = {c: 0 for c in feature_cols}
    scores["MGC_ptosis"] = 1
    scores["MGADL_ptosis"] = 1
    return scores


def build_preset_severe(feature_cols, item_max):
    return {c: min(3, item_max[c]) for c in feature_cols}


def build_preset_zero(feature_cols):
    return {c: 0 for c in feature_cols}


# =====================================================================================
# CSS — クリーン・クリニカル（ライトテーマ）
# =====================================================================================
ACCENT = "#0f766e"       # clinical teal
ACCENT_SOFT = "#f0faf8"
MM_COLOR = "#15803d"     # clinical green
MM_SOFT = "#f0faf4"
NONMM_COLOR = "#b45309"  # muted amber
NONMM_SOFT = "#fdf6ec"
INK = "#1e293b"
MUTED = "#64748b"
LINE = "#e2e8f0"

CUSTOM_CSS = f"""
<style>
#MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{
    display: none !important;
}}
div[data-testid="stStatusWidget"] {{ display: none !important; }}

html, body, [data-testid="stAppViewContainer"] {{
    background: #f5f7fa !important;
    color: {INK};
    font-family: -apple-system, "Segoe UI", Roboto, "Hiragino Kaku Gothic ProN", "Helvetica Neue", sans-serif;
}}
[data-testid="stAppViewContainer"] > .main {{
    background: transparent !important;
}}
[data-testid="stHeader"] {{ background: transparent !important; }}

.block-container {{
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}}

/* ---------- Masthead ---------- */
.masthead {{
    padding: 1.1rem 1.5rem;
    background: #ffffff;
    border: 1px solid {LINE};
    border-left: 5px solid {ACCENT};
    border-radius: 10px;
    margin-bottom: 1.4rem;
    box-shadow: 0 1px 3px rgba(16,24,40,0.06);
}}
.masthead-title {{
    font-size: 1.7rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    margin: 0;
    line-height: 1.25;
    color: {INK};
}}
.masthead-sub {{
    margin-top: 0.25rem;
    font-size: 0.98rem;
    color: {MUTED};
    font-weight: 400;
}}
.masthead-tags {{ display: flex; gap: 0.5rem; margin-top: 0.7rem; flex-wrap: wrap; }}
.masthead-tag {{
    padding: 0.2rem 0.65rem;
    border-radius: 4px;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    background: {ACCENT_SOFT};
    border: 1px solid #cfe8e4;
    color: {ACCENT};
}}

/* ---------- Card ---------- */
.clin-card {{
    background: #ffffff;
    border: 1px solid {LINE};
    border-radius: 10px;
    padding: 1.2rem 1.4rem 1.05rem 1.4rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 1px 3px rgba(16,24,40,0.06);
}}
.clin-card h4 {{
    margin-top: 0;
    margin-bottom: 0.9rem;
    font-size: 0.98rem;
    font-weight: 700;
    color: {INK};
    padding-bottom: 0.55rem;
    border-bottom: 1px solid {LINE};
}}

.section-title {{
    font-size: 1.15rem;
    font-weight: 700;
    margin: 1.5rem 0 0.8rem 0;
    color: {INK};
    padding-left: 0.7rem;
    border-left: 4px solid {ACCENT};
}}

/* ---------- Preset / secondary buttons ---------- */
div[data-testid="column"] .stButton > button,
.stButton > button[kind="secondary"] {{
    border-radius: 6px !important;
    font-weight: 600 !important;
    border: 1px solid #cbd5e1 !important;
    background: #ffffff !important;
    color: {INK} !important;
    transition: all 0.12s ease;
    padding: 0.5rem 0.6rem !important;
}}
div[data-testid="column"] .stButton > button:hover,
.stButton > button[kind="secondary"]:hover {{
    border-color: {ACCENT} !important;
    background: {ACCENT_SOFT} !important;
    color: {ACCENT} !important;
}}

/* primary CTA button (判定する) */
.stButton > button[kind="primary"] {{
    background: {ACCENT};
    border: 1px solid {ACCENT};
    color: #ffffff;
    font-weight: 700;
    font-size: 1.02rem;
    padding: 0.7rem 1rem;
    border-radius: 8px;
    letter-spacing: 0.01em;
    box-shadow: none;
}}
.stButton > button[kind="primary"]:hover {{
    background: #0d5f58;
    border-color: #0d5f58;
    color: #ffffff;
}}
.stButton > button[kind="primary"] p {{ color: #ffffff !important; }}

/* ---------- Sliders ---------- */
[data-testid="stSlider"] label p {{
    font-weight: 600;
    color: {INK};
    font-size: 0.88rem;
}}
div[data-baseweb="slider"] > div > div {{ background: #dbe3ec !important; }}
div[data-baseweb="slider"] div[role="slider"] {{
    background: {ACCENT} !important;
    box-shadow: none !important;
    border: 2px solid #ffffff;
}}

/* ---------- Verdict report card ---------- */
.verdict-card {{
    background: #ffffff;
    border: 1px solid {LINE};
    border-radius: 10px;
    padding: 1.1rem 1.5rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 1px 3px rgba(16,24,40,0.06);
    display: flex;
    align-items: center;
    gap: 1.4rem;
}}
.verdict-card.mm {{ border-left: 6px solid {MM_COLOR}; }}
.verdict-card.nonmm {{ border-left: 6px solid {NONMM_COLOR}; }}
.verdict-label {{
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: 0.01em;
    line-height: 1;
}}
.verdict-label.mm {{ color: {MM_COLOR}; }}
.verdict-label.nonmm {{ color: {NONMM_COLOR}; }}
.verdict-text {{ flex: 1; }}
.verdict-desc {{ font-size: 1rem; font-weight: 600; color: {INK}; margin-bottom: 0.15rem; }}
.verdict-meta {{ font-size: 0.85rem; color: {MUTED}; }}

/* ---------- Model agreement table ---------- */
.model-table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
.model-table th {{
    text-align: left;
    padding: 0.5rem 0.6rem;
    color: {MUTED};
    font-weight: 600;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    border-bottom: 2px solid {LINE};
}}
.model-table td {{
    padding: 0.55rem 0.6rem;
    border-bottom: 1px solid {LINE};
    color: {INK};
}}
.model-table tr:last-child td {{ border-bottom: none; }}
.model-table td.num {{ font-weight: 700; text-align: right; font-variant-numeric: tabular-nums; }}
.tag-pill {{
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.15rem 0.55rem;
    border-radius: 4px;
}}
.tag-mm {{ color: {MM_COLOR}; background: {MM_SOFT}; border: 1px solid #bfe3cd; }}
.tag-nonmm {{ color: {NONMM_COLOR}; background: {NONMM_SOFT}; border: 1px solid #edd6b0; }}

.footer-note {{
    text-align: center;
    color: {MUTED};
    font-size: 0.8rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid {LINE};
}}
</style>
"""


# =====================================================================================
# UI helpers
# =====================================================================================
def gauge_figure(prob: float, is_mm: bool) -> go.Figure:
    color = MM_COLOR if is_mm else NONMM_COLOR
    fig = go.Figure(
        go.Pie(
            values=[prob, 1 - prob],
            hole=0.78,
            sort=False,
            direction="clockwise",
            marker=dict(colors=[color, "#eef1f5"], line=dict(color="#ffffff", width=1)),
            textinfo="none",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_annotation(
        text=f"<span style='font-size:44px;font-weight:800;color:{INK}'>{prob*100:.1f}%</span>",
        x=0.5, y=0.56, showarrow=False, xref="paper", yref="paper",
    )
    fig.add_annotation(
        text=f"<span style='font-size:12px;color:{MUTED};font-weight:600;letter-spacing:0.05em'>ENSEMBLE PROBABILITY</span>",
        x=0.5, y=0.40, showarrow=False, xref="paper", yref="paper",
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        height=280,
        annotations=fig.layout.annotations,
    )
    return fig


def radar_figure(patient, nonmm_mean):
    axes = MODULES + [MODULES[0]]
    axes_ja = [MODULE_LABELS_JA[m] for m in axes]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=nonmm_mean + [nonmm_mean[0]],
            theta=axes_ja,
            fill="toself",
            name="non-MM群 平均",
            line=dict(color="#94a3b8", width=1.5, dash="dash"),
            fillcolor="rgba(148,163,184,0.15)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=patient + [patient[0]],
            theta=axes_ja,
            fill="toself",
            name="この患者",
            line=dict(color=ACCENT, width=2.5),
            fillcolor="rgba(15,118,110,0.15)",
        )
    )
    rmax = max(max(nonmm_mean), max(patient)) * 1.2 or 0.1
    fig.update_layout(
        template="plotly_white",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, rmax], showticklabels=True,
                tickfont=dict(color=MUTED, size=9),
                gridcolor="#e5eaf0",
                linecolor="#e5eaf0",
            ),
            angularaxis=dict(
                tickfont=dict(color=INK, size=13, family="-apple-system, Segoe UI"),
                gridcolor="#e5eaf0",
                linecolor="#e5eaf0",
            ),
        ),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="top", y=-0.16, x=0.5, xanchor="center",
            font=dict(color=INK, size=12),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK),
        height=420,
        margin=dict(t=30, b=80, l=50, r=50),
    )
    return fig


def module_bar_figure(patient):
    labels = [MODULE_LABELS_JA[m] for m in MODULES]
    fig = go.Figure(
        go.Bar(
            x=patient,
            y=labels,
            orientation="h",
            marker=dict(
                color=patient,
                colorscale=[[0, "#7fc8bf"], [1, ACCENT]],
                line=dict(width=0),
            ),
            text=[f"{v:.3f}" for v in patient],
            textposition="outside",
            textfont=dict(color=INK, size=12, family="-apple-system"),
        )
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=210,
        margin=dict(t=10, b=10, l=10, r=30),
        xaxis=dict(showgrid=True, gridcolor="#e5eaf0", color=MUTED, zeroline=False),
        yaxis=dict(color=INK, tickfont=dict(size=13)),
        font=dict(color=INK),
    )
    return fig


# =====================================================================================
# Main
# =====================================================================================
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    feature_cols, scaler, nmf, cutoffs, folds, item_max, group_stats = load_artifacts()

    # ---- session_state 初期化 --------------------------------------------------
    if "scores_initialized" not in st.session_state:
        for c in feature_cols:
            st.session_state[c] = 0
        st.session_state.scores_initialized = True
    if "show_result" not in st.session_state:
        st.session_state.show_result = False

    # ---- マストヘッド --------------------------------------------------------------
    st.markdown(
        """
        <div class="masthead">
            <div class="masthead-title">MG 最小症状発現（MM）予測</div>
            <div class="masthead-sub">MGC・MG-ADL・MG-QOL15 スコアからMM到達を推定します</div>
            <div class="masthead-tags">
                <span class="masthead-tag">NMF 4モジュール分解</span>
                <span class="masthead-tag">4モデル アンサンブル</span>
                <span class="masthead-tag">MGC ・ MG-ADL ・ MG-QOL15</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- プリセットボタン ----------------------------------------------------------
    st.markdown('<div class="section-title">デモプリセット</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        if st.button("軽症例（MM想定）", use_container_width=True):
            preset = build_preset_mild(feature_cols)
            for k, v in preset.items():
                st.session_state[k] = v
            st.session_state.show_result = False
            st.rerun()
    with p2:
        if st.button("重症例（non-MM想定）", use_container_width=True):
            preset = build_preset_severe(feature_cols, item_max)
            for k, v in preset.items():
                st.session_state[k] = v
            st.session_state.show_result = False
            st.rerun()
    with p3:
        if st.button("リセット", use_container_width=True):
            preset = build_preset_zero(feature_cols)
            for k, v in preset.items():
                st.session_state[k] = v
            st.session_state.show_result = False
            st.rerun()

    # ---- 入力フォーム --------------------------------------------------------------
    st.markdown('<div class="section-title">患者スコア入力</div>', unsafe_allow_html=True)
    scores = {}
    cols = st.columns(3)
    for col, (group, items) in zip(cols, ITEM_LABELS.items()):
        with col:
            st.markdown(
                f'<div class="clin-card"><h4>{group}　<span style="color:{MUTED};font-weight:500;font-size:0.82rem">{GROUP_LABEL_EN[group]}</span></h4>',
                unsafe_allow_html=True,
            )
            for key, label in items.items():
                scores[key] = st.slider(
                    label, 0, item_max[key], st.session_state.get(key, 0), key=key
                )
            st.markdown("</div>", unsafe_allow_html=True)

    run = st.button("判定する", type="primary", use_container_width=True)
    if run:
        st.session_state.show_result = True

    if not st.session_state.show_result:
        st.markdown(
            '<div class="footer-note">研究目的のデモです。臨床判断は医師が行ってください。</div>',
            unsafe_allow_html=True,
        )
        return

    # ---- 予測 -----------------------------------------------------------------
    W, results = predict(scores, feature_cols, scaler, nmf, cutoffs, folds)
    ensemble_prob = float(np.mean([r["prob"] for r in results.values()]))
    votes = sum(r["label"] for r in results.values())
    ensemble_mm = votes >= 2

    # ---- 結果表示 ----------------------------------------------------------------
    st.markdown('<div class="section-title">判定結果</div>', unsafe_allow_html=True)

    verdict_class = "mm" if ensemble_mm else "nonmm"
    verdict_text = "MM" if ensemble_mm else "non-MM"
    verdict_desc = "治療反応 良好が示唆されます" if ensemble_mm else "治療反応 不良が示唆されます"
    st.markdown(
        f"""
        <div class="verdict-card {verdict_class}">
            <div class="verdict-label {verdict_class}">{verdict_text}</div>
            <div class="verdict-text">
                <div class="verdict-desc">{verdict_desc}</div>
                <div class="verdict-meta">MM判定したモデル数: {votes} / 4　｜　アンサンブル確率: {ensemble_prob:.1%}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rc1, rc2 = st.columns([1, 1.15])
    with rc1:
        st.markdown('<div class="clin-card"><h4>アンサンブル確率</h4>', unsafe_allow_html=True)
        st.plotly_chart(gauge_figure(ensemble_prob, ensemble_mm), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="clin-card"><h4>4モデルの一致</h4>', unsafe_allow_html=True)
        rows_html = ""
        for name, r in results.items():
            pct = r["prob"] * 100
            tag_class = "tag-mm" if r["label"] else "tag-nonmm"
            tag_text = "MM" if r["label"] else "non-MM"
            rows_html += (
                f'<tr><td>{name}</td><td class="num">{pct:.1f}%</td>'
                f'<td><span class="tag-pill {tag_class}">{tag_text}</span></td></tr>'
            )
        table_html = (
            '<table class="model-table"><thead><tr><th>モデル</th>'
            '<th style="text-align:right">確率</th><th>判定</th></tr></thead>'
            f'<tbody>{rows_html}</tbody></table>'
        )
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with rc2:
        st.markdown('<div class="clin-card"><h4>NMFモジュール比較（ダイヤチャート）</h4>', unsafe_allow_html=True)
        nonmm_mean = [group_stats["nonMM_mean"][m] for m in MODULES]
        patient = [float(v) for v in W]
        st.plotly_chart(radar_figure(patient, nonmm_mean), use_container_width=True, config={"displayModeBar": False})
        st.caption("実線（緑がかった青）：この患者　　破線（灰色）：non-MM群 平均　外側ほど症状負荷が大きい")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="clin-card"><h4>モジュール別 寄与度</h4>', unsafe_allow_html=True)
    st.plotly_chart(module_bar_figure(patient), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="footer-note">研究目的です。臨床判断は医師が行ってください。'
        '判定は各モデルのROC最適カットオフ（Youden指数）に基づきます。</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
