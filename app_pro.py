"""
MG MM Predictor — Pro (dark premium UI)

判定ロジックは app_simple.py と完全に同一（数値・アンサンブル手法とも）。
デザインのみを大幅に強化したスクリーンショット用アプリ。

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
    page_icon="🧬",
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

GROUP_ICON = {"MGC": "👁️", "MGADL": "🧍", "MGQOL15": "💬"}


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
# CSS — ダーク・プレミアム・グラスモーフィズム
# =====================================================================================
CUSTOM_CSS = """
<style>
#MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
    display: none !important;
}
div[data-testid="stStatusWidget"] { display: none !important; }

html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 120% 60% at 50% -10%, #17203a 0%, #0b1020 45%, #070a14 100%) !important;
    color: #e6ebf5;
    font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
}
[data-testid="stAppViewContainer"] > .main {
    background: transparent !important;
}
[data-testid="stHeader"] { background: transparent !important; }

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

/* ---------- Hero ---------- */
.hero-wrap { text-align: center; padding: 1.2rem 0 1.6rem 0; }
.hero-title {
    font-size: 3.6rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.05;
    background: linear-gradient(90deg, #2dd4bf 0%, #22d3ee 42%, #8b5cf6 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 26px rgba(34, 211, 238, 0.35));
}
.hero-sub-jp {
    margin-top: 0.35rem;
    font-size: 1.3rem;
    font-weight: 700;
    color: #cdd6e8;
}
.hero-catch {
    margin-top: 0.6rem;
    font-size: 1.05rem;
    color: #8b95ad;
    font-weight: 500;
}
.hero-pills { display: flex; justify-content: center; gap: 0.6rem; margin-top: 1rem; flex-wrap: wrap; }
.hero-pill {
    padding: 0.32rem 0.9rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    background: rgba(45, 212, 191, 0.10);
    border: 1px solid rgba(45, 212, 191, 0.35);
    color: #5eead4;
}

/* ---------- Glass card ---------- */
.glass-card {
    background: rgba(255,255,255,0.045);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 18px;
    padding: 1.35rem 1.5rem 1.1rem 1.5rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
}
.glass-card h4 {
    margin-top: 0;
    font-size: 1.05rem;
    font-weight: 800;
    color: #f1f5f9;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.card-icon-badge {
    display: inline-block;
    font-size: 1.1rem;
}

.section-title {
    font-size: 1.5rem;
    font-weight: 900;
    margin: 1.6rem 0 0.9rem 0;
    background: linear-gradient(90deg, #22d3ee, #8b5cf6);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}

/* ---------- Preset buttons (secondary) ---------- */
div[data-testid="column"] .stButton > button,
.stButton > button[kind="secondary"] {
    border-radius: 12px !important;
    font-weight: 700 !important;
    border: 1px solid rgba(255,255,255,0.16) !important;
    background: rgba(255,255,255,0.06) !important;
    color: #e6ebf5 !important;
    transition: all 0.15s ease;
    padding: 0.55rem 0.6rem !important;
}
div[data-testid="column"] .stButton > button:hover,
.stButton > button[kind="secondary"]:hover {
    border-color: rgba(45, 212, 191, 0.55) !important;
    background: rgba(45, 212, 191, 0.12) !important;
    color: #5eead4 !important;
    transform: translateY(-1px);
}

/* primary CTA button (判定を実行) */
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #2dd4bf, #22d3ee 55%, #8b5cf6);
    border: none;
    color: #04121b;
    font-weight: 900;
    font-size: 1.15rem;
    padding: 0.85rem 1rem;
    border-radius: 14px;
    box-shadow: 0 0 28px rgba(34, 211, 238, 0.45), 0 8px 20px rgba(139, 92, 246, 0.25);
    letter-spacing: 0.02em;
}
.stButton > button[kind="primary"]:hover {
    filter: brightness(1.08);
    box-shadow: 0 0 40px rgba(34, 211, 238, 0.65), 0 8px 24px rgba(139, 92, 246, 0.35);
}

/* ---------- Sliders ---------- */
[data-testid="stSlider"] label p {
    font-weight: 700;
    color: #c7cfe2;
    font-size: 0.92rem;
}
div[data-baseweb="slider"] > div > div { background: rgba(255,255,255,0.10) !important; }
div[data-baseweb="slider"] div[role="slider"] {
    background: #2dd4bf !important;
    box-shadow: 0 0 10px rgba(45, 212, 191, 0.8);
}

/* ---------- Verdict badge ---------- */
.verdict-wrap { text-align: center; padding: 0.4rem 0 0.2rem 0; }
.verdict-badge {
    display: inline-block;
    font-size: 3.6rem;
    font-weight: 900;
    letter-spacing: 0.02em;
    padding: 0.35rem 2.4rem;
    border-radius: 22px;
    margin-bottom: 0.4rem;
}
.verdict-mm {
    color: #eafff9;
    background: linear-gradient(135deg, rgba(45,212,191,0.28), rgba(34,211,238,0.18));
    border: 2px solid rgba(45,212,191,0.75);
    box-shadow: 0 0 55px rgba(45, 212, 191, 0.55), inset 0 0 25px rgba(45,212,191,0.15);
    text-shadow: 0 0 24px rgba(45,212,191,0.9);
}
.verdict-nonmm {
    color: #fff1f2;
    background: linear-gradient(135deg, rgba(251,113,133,0.26), rgba(244,63,94,0.15));
    border: 2px solid rgba(251,113,133,0.75);
    box-shadow: 0 0 55px rgba(251, 113, 133, 0.5), inset 0 0 25px rgba(251,113,133,0.15);
    text-shadow: 0 0 24px rgba(251,113,133,0.9);
}
.verdict-caption { color: #93a0ba; font-size: 0.95rem; font-weight: 600; margin-top: 0.2rem;}

/* ---------- Model pills ---------- */
.model-pill-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.55rem 0.9rem;
    margin-bottom: 0.55rem;
    border-radius: 12px;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
}
.model-pill-name { flex: 0 0 150px; font-weight: 700; font-size: 0.88rem; color: #d5dcee; }
.model-pill-bar-bg { flex: 1; height: 10px; background: rgba(255,255,255,0.08); border-radius: 999px; overflow: hidden; }
.model-pill-bar-fill { height: 100%; border-radius: 999px; }
.model-pill-fill-mm { background: linear-gradient(90deg, #2dd4bf, #22d3ee); box-shadow: 0 0 10px rgba(45,212,191,0.7); }
.model-pill-fill-nonmm { background: linear-gradient(90deg, #fb7185, #f43f5e); box-shadow: 0 0 10px rgba(251,113,133,0.7); }
.model-pill-pct { flex: 0 0 62px; text-align: right; font-weight: 800; font-size: 0.95rem; }
.model-pill-tag { flex: 0 0 76px; text-align: center; font-size: 0.72rem; font-weight: 800; padding: 0.15rem 0.5rem; border-radius: 999px; }
.tag-mm { color: #04231d; background: #2dd4bf; }
.tag-nonmm { color: #2a0a0f; background: #fb7185; }

.footer-note {
    text-align: center;
    color: #5c6885;
    font-size: 0.82rem;
    margin-top: 2.2rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255,255,255,0.06);
}
</style>
"""


# =====================================================================================
# UI helpers
# =====================================================================================
def gauge_figure(prob: float, is_mm: bool) -> go.Figure:
    color = "#2dd4bf" if is_mm else "#fb7185"
    glow = "rgba(45,212,191,0.35)" if is_mm else "rgba(251,113,133,0.35)"
    fig = go.Figure(
        go.Pie(
            values=[prob, 1 - prob],
            hole=0.78,
            sort=False,
            direction="clockwise",
            marker=dict(colors=[color, "rgba(255,255,255,0.07)"], line=dict(color="rgba(0,0,0,0)", width=0)),
            textinfo="none",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_annotation(
        text=f"<span style='font-size:52px;font-weight:900;color:{color}'>{prob*100:.1f}%</span>",
        x=0.5, y=0.56, showarrow=False, xref="paper", yref="paper",
    )
    fig.add_annotation(
        text="<span style='font-size:14px;color:#93a0ba;font-weight:700;letter-spacing:0.05em'>ENSEMBLE PROBABILITY</span>",
        x=0.5, y=0.40, showarrow=False, xref="paper", yref="paper",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        height=300,
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
            line=dict(color="#fb7185", width=2),
            fillcolor="rgba(251,113,133,0.22)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=patient + [patient[0]],
            theta=axes_ja,
            fill="toself",
            name="この患者",
            line=dict(color="#38bdf8", width=3),
            fillcolor="rgba(56,189,248,0.28)",
        )
    )
    rmax = max(max(nonmm_mean), max(patient)) * 1.2 or 0.1
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, rmax], showticklabels=True,
                tickfont=dict(color="#7a869e", size=9),
                gridcolor="rgba(255,255,255,0.12)",
                linecolor="rgba(255,255,255,0.12)",
            ),
            angularaxis=dict(
                tickfont=dict(color="#dbe2f2", size=13, family="-apple-system, Segoe UI"),
                gridcolor="rgba(255,255,255,0.12)",
                linecolor="rgba(255,255,255,0.12)",
            ),
        ),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="top", y=-0.16, x=0.5, xanchor="center",
            font=dict(color="#cdd6e8", size=12),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6ebf5"),
        height=440,
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
                colorscale=[[0, "#22d3ee"], [1, "#8b5cf6"]],
                line=dict(width=0),
            ),
            text=[f"{v:.3f}" for v in patient],
            textposition="outside",
            textfont=dict(color="#e6ebf5", size=12, family="-apple-system"),
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(t=10, b=10, l=10, r=30),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", color="#93a0ba", zeroline=False),
        yaxis=dict(color="#dbe2f2", tickfont=dict(size=13)),
        font=dict(color="#e6ebf5"),
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

    # ---- ヒーロー ----------------------------------------------------------------
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-title">MG MM Predictor</div>
            <div class="hero-sub-jp">重症筋無力症 MM到達 予測AI</div>
            <div class="hero-catch">3つの臨床スコアだけで、MM到達を予測 — NMF × アンサンブル機械学習</div>
            <div class="hero-pills">
                <span class="hero-pill">NMF 4-MODULE</span>
                <span class="hero-pill">4 MODEL ENSEMBLE</span>
                <span class="hero-pill">MGC · MG-ADL · MGQOL15</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- プリセットボタン ----------------------------------------------------------
    st.markdown('<div class="section-title">⚡ デモプリセット</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        if st.button("🟢 軽症サンプル（MM想定）", use_container_width=True):
            preset = build_preset_mild(feature_cols)
            for k, v in preset.items():
                st.session_state[k] = v
            st.session_state.show_result = False
            st.rerun()
    with p2:
        if st.button("🔴 重症サンプル（non-MM想定）", use_container_width=True):
            preset = build_preset_severe(feature_cols, item_max)
            for k, v in preset.items():
                st.session_state[k] = v
            st.session_state.show_result = False
            st.rerun()
    with p3:
        if st.button("↺ リセット", use_container_width=True):
            preset = build_preset_zero(feature_cols)
            for k, v in preset.items():
                st.session_state[k] = v
            st.session_state.show_result = False
            st.rerun()

    # ---- 入力フォーム --------------------------------------------------------------
    st.markdown('<div class="section-title">📝 患者スコア入力</div>', unsafe_allow_html=True)
    scores = {}
    cols = st.columns(3)
    for col, (group, items) in zip(cols, ITEM_LABELS.items()):
        with col:
            st.markdown(
                f'<div class="glass-card"><h4><span class="card-icon-badge">{GROUP_ICON[group]}</span>{group}</h4>',
                unsafe_allow_html=True,
            )
            for key, label in items.items():
                scores[key] = st.slider(
                    label, 0, item_max[key], st.session_state.get(key, 0), key=key
                )
            st.markdown("</div>", unsafe_allow_html=True)

    run = st.button("🚀 判定を実行", type="primary", use_container_width=True)
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
    st.markdown('<div class="section-title">🔍 判定結果</div>', unsafe_allow_html=True)

    verdict_class = "verdict-mm" if ensemble_mm else "verdict-nonmm"
    verdict_text = "MM" if ensemble_mm else "non-MM"
    st.markdown(
        f"""
        <div class="verdict-wrap">
            <div class="verdict-badge {verdict_class}">{verdict_text}</div>
            <div class="verdict-caption">MM判定したモデル数: {votes} / 4</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rc1, rc2 = st.columns([1, 1.15])
    with rc1:
        st.markdown('<div class="glass-card"><h4>🎯 アンサンブル確率</h4>', unsafe_allow_html=True)
        st.plotly_chart(gauge_figure(ensemble_prob, ensemble_mm), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="glass-card"><h4>🧪 4モデルの合議</h4>', unsafe_allow_html=True)
        pill_html = ""
        for name, r in results.items():
            pct = r["prob"] * 100
            tag_class = "tag-mm" if r["label"] else "tag-nonmm"
            fill_class = "model-pill-fill-mm" if r["label"] else "model-pill-fill-nonmm"
            pill_html += f"""
            <div class="model-pill-row">
                <div class="model-pill-name">{name}</div>
                <div class="model-pill-bar-bg"><div class="model-pill-bar-fill {fill_class}" style="width:{pct:.1f}%"></div></div>
                <div class="model-pill-pct">{pct:.1f}%</div>
                <div class="model-pill-tag {tag_class}">{'MM' if r['label'] else 'non-MM'}</div>
            </div>
            """
        st.markdown(pill_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with rc2:
        st.markdown('<div class="glass-card"><h4>💠 NMFモジュール比較（ダイヤチャート）</h4>', unsafe_allow_html=True)
        nonmm_mean = [group_stats["nonMM_mean"][m] for m in MODULES]
        patient = [float(v) for v in W]
        st.plotly_chart(radar_figure(patient, nonmm_mean), use_container_width=True, config={"displayModeBar": False})
        st.caption("青：この患者　　赤：non-MM群 平均　外側ほど症状負荷が大きい")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card"><h4>📊 モジュール別 寄与度</h4>', unsafe_allow_html=True)
    st.plotly_chart(module_bar_figure(patient), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="footer-note">研究目的です。臨床判断は医師が行ってください。'
        '判定は各モデルのROC最適カットオフ（Youden指数）に基づきます。</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
