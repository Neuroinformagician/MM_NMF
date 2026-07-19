"""
MG MM判定システム（シンプル版）

患者のスコア（MGC / MGADL / MGQOL15、計33項目）を入力すると、
学習済みの MinMaxScaler → NMF → 機械学習モデル（4種×5fold）を通して、
治療反応性「MM or better」を確率つきで判定します。

起動:
    streamlit run app_simple.py
"""

import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="MG MM判定システム", page_icon="🏥", layout="wide")

# --- モデルの表示名 と fold ファイルの prefix / cutoff キー の対応 ------------
MODELS = {
    "svm": ("SVM", "SVM"),
    "logistic_regression": ("ロジスティック回帰", "Logistic Regression"),
    "random_forest": ("ランダムフォレスト", "Random Forest"),
    "naive_bayes": ("ナイーブベイズ", "GaussianNB"),
}
MODULES = ["module QOL", "module Diplopia", "module Ptosis", "module Systemic"]

# --- 33項目の日本語ラベル（feature_cols.pkl と同じ並び順） -------------------
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


@st.cache_resource
def load_artifacts():
    """学習済みモデル一式を読み込む（初回のみ実行しキャッシュ）。"""
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

    # スライダーの上限は学習データで観測された各項目の最大値を使う
    item_max = {col: int(round(m)) for col, m in zip(feature_cols, scaler.data_max_)}
    return feature_cols, scaler, nmf, cutoffs, folds, item_max, group_stats


def predict(scores, feature_cols, scaler, nmf, cutoffs, folds):
    """33項目のスコア dict から、モジュール値と各モデルの確率・判定を返す。"""
    x = pd.DataFrame([[scores[c] for c in feature_cols]], columns=feature_cols)
    x_scaled = scaler.transform(x)             # 学習時に fit したスケーラーで正規化
    W = nmf.transform(x_scaled)                # 33項目 → 4モジュール
    W_df = pd.DataFrame(W, columns=MODULES)

    results = {}
    for prefix, (disp, cut_key) in MODELS.items():
        prob = float(np.mean([m.predict_proba(W_df)[0, 1] for m in folds[prefix]]))
        results[disp] = {"prob": prob, "label": prob >= cutoffs[cut_key]}
    return W[0], results


def main():
    st.title("🏥 重症筋無力症（MG）MM判定システム")
    st.caption(
        "MGC・MGADL・MGQOL15 のスコアを入力すると、NMF＋機械学習で治療反応性"
        "（MM or better）を予測します。"
    )

    feature_cols, scaler, nmf, cutoffs, folds, item_max, group_stats = load_artifacts()

    # ---- 入力フォーム -------------------------------------------------------
    st.markdown("### 📝 患者スコア入力")
    scores = {}
    cols = st.columns(3)
    for col, (group, items) in zip(cols, ITEM_LABELS.items()):
        with col:
            st.markdown(f"#### {group}")
            for key, label in items.items():
                scores[key] = st.slider(label, 0, item_max[key], 0, key=key)

    if not st.button("🚀 判定を実行", type="primary", use_container_width=True):
        return

    # ---- 予測 ---------------------------------------------------------------
    W, results = predict(scores, feature_cols, scaler, nmf, cutoffs, folds)
    ensemble_prob = float(np.mean([r["prob"] for r in results.values()]))
    votes = sum(r["label"] for r in results.values())
    ensemble_mm = votes >= 2  # 4モデル中2つ以上が MM なら MM or better

    # ---- 結果表示 -----------------------------------------------------------
    st.markdown("---")
    st.markdown("## 🔍 判定結果")

    c1, c2 = st.columns([1, 1])
    with c1:
        if ensemble_mm:
            st.success("### ✅ MM or better（治療反応性 良好）")
        else:
            st.warning("### ⚠️ non-MM（治療反応性 不良）")
        st.metric("平均確率（MM or better）", f"{ensemble_prob:.1%}")
        st.caption(f"MM判定したモデル数: {votes} / 4")

        st.markdown("#### モデル別の判定")
        table = pd.DataFrame(
            {
                "モデル": list(results.keys()),
                "確率": [f"{r['prob']:.1%}" for r in results.values()],
                "判定": ["MM or better" if r["label"] else "non-MM" for r in results.values()],
            }
        )
        st.dataframe(table, hide_index=True, use_container_width=True)

    with c2:
        st.markdown("#### NMFモジュール値（ダイヤチャート）")
        nonmm_mean = [group_stats["nonMM_mean"][m] for m in MODULES]
        patient = [float(v) for v in W]

        # レーダーチャートは始点に戻すため各系列の先頭要素を末尾にも追加する
        axes = MODULES + [MODULES[0]]
        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=nonmm_mean + [nonmm_mean[0]],
                theta=axes,
                fill="toself",
                name="non-MM群 平均",
                line=dict(color="#1f77b4"),
                fillcolor="rgba(31,119,180,0.25)",
            )
        )
        fig.add_trace(
            go.Scatterpolar(
                r=patient + [patient[0]],
                theta=axes,
                fill="toself",
                name="この患者",
                line=dict(color="#d62728", width=2),
                fillcolor="rgba(214,39,40,0.20)",
            )
        )
        rmax = max(max(nonmm_mean), max(patient)) * 1.15 or 0.1
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, rmax])),
            showlegend=True,
            height=380,
            margin=dict(t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("青の面：non-MM群の平均、赤の面：この患者。外側ほど症状負荷が大きい。")

    st.info(
        "判定は各モデルのROC最適カットオフ（Youden指数）に基づきます。"
        "本システムは研究目的であり、臨床判断は医師が行ってください。"
    )


if __name__ == "__main__":
    main()
