import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from typing import Dict, List
from scipy.special import expit

st.set_page_config(
    page_title="MG MM判定システム",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# グリッドUIのCSS
st.markdown("""
<style>
    .block-container {
        padding: 0.5rem;
        max-width: 100%;
    }
    
    .section-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        padding: 8px 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-weight: bold;
    }
    
    .result-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 12px;
        padding: 1rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    h1 { font-size: 1.8rem; margin: 0.5rem 0; }
    h2 { font-size: 1.4rem; margin: 0.4rem 0; }
    h3 { font-size: 1.2rem; margin: 0.3rem 0; }
    
    .stButton > button {
        width: 100%;
        padding: 8px;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """学習済みモデル・前処理・付帯情報を読み込む"""
    models = {}
    try:
        # NMF
        with open('./out/nmf_model.pkl', 'rb') as f:
            models['nmf'] = pickle.load(f)

        # 各foldモデル
        models_cv = {'LR': [], 'SVM': [], 'RF': [], 'NB': []}
        for fold in range(5):
            with open(f'./out/logistic_regression_fold_{fold}.pkl', 'rb') as f:
                models_cv['LR'].append(pickle.load(f))
            with open(f'./out/svm_fold_{fold}.pkl', 'rb') as f:
                models_cv['SVM'].append(pickle.load(f))
            with open(f'./out/random_forest_fold_{fold}.pkl', 'rb') as f:
                models_cv['RF'].append(pickle.load(f))
            with open(f'./out/naive_bayes_fold_{fold}.pkl', 'rb') as f:
                models_cv['NB'].append(pickle.load(f))
        models['models_cv'] = models_cv

        # 群統計
        with open('./out/group_statistics.pkl', 'rb') as f:
            models['group_stats'] = pickle.load(f)

        # ★ 学習時の MinMaxScaler（fit禁止・transform専用）
        with open('./out/minmax_scaler.pkl', 'rb') as f:
            models['scaler'] = pickle.load(f)

        # ★ 任意: 学習時の列順と最終しきい値
        try:
            with open('./out/feature_cols.pkl', 'rb') as f:
                models['feature_cols'] = pickle.load(f)
        except FileNotFoundError:
            models['feature_cols'] = None
        try:
            with open('./out/thresholds.pkl', 'rb') as f:
                models['thresholds'] = pickle.load(f)  # 例: {"ensemble": 0.47}
        except FileNotFoundError:
            models['thresholds'] = None

        return models
    except FileNotFoundError as e:
        st.error(f"モデルファイルが見つかりません: {e}。'out' ディレクトリを確認してください。")
        return None

def predict_case(
    case_input,
    feature_cols: List[str],
    scaler,
    nmf,
    models_cv: Dict[str, List],
    thresholds: Dict[str, float] = None,
    ensemble_rule: str = "soft"
) -> Dict:
    # 入力→1行DF→列順
    X = pd.DataFrame([case_input]).reindex(columns=feature_cols)

    # ★ 値域チェック（入力ミス防止）
    bounds = {}
    for c in feature_cols:
        if c.startswith("MGADL_"): bounds[c] = (0, 3)
        elif c.startswith("MGC_"):  bounds[c] = (0, 5)
        elif c.startswith("MGQOL"): bounds[c] = (0, 4)
    bad = [f"{c}={X.iat[0, i]}" for i, c in enumerate(feature_cols)
           if c in bounds and not (bounds[c][0] <= float(X.iat[0, i]) <= bounds[c][1])]
    if bad:
        raise ValueError("スコアの範囲外があります: " + ", ".join(bad))

    # ★ 学習時scalerで transform のみ（fit/fit_transform禁止）
    Xs = scaler.transform(X)

    # NMF transform
    W = nmf.transform(Xs)  # (1, n_components)

    # 各モデル（fold平均）確率
    per_model_prob = {}
    for name, estimators in models_cv.items():
        fold_ps = []
        for est in estimators:
            if hasattr(est, "predict_proba"):
                p = est.predict_proba(W)[:, 1]
            elif hasattr(est, "decision_function"):
                p = expit(est.decision_function(W))
            else:
                raise ValueError(f"{name} に predict_proba/decision_function がありません。")
            fold_ps.append(p)
        per_model_prob[name] = float(np.mean(np.vstack(fold_ps), axis=0))

    # アンサンブル
    if ensemble_rule == "soft":
        p_ens = float(np.mean(list(per_model_prob.values())))
        thr = thresholds.get("ensemble", 0.5) if thresholds else 0.5
        yhat = int(p_ens >= thr)
        prob_block = {"per_model": per_model_prob, "ensemble_soft": p_ens}
    else:
        votes = []
        for name, p in per_model_prob.items():
            thr_m = thresholds.get(name, 0.5) if thresholds else 0.5
            votes.append(int(p >= thr_m))
        vote_rate = float(np.mean(votes))
        yhat = int(vote_rate >= 0.5)
        thr = 0.5
        prob_block = {"per_model": per_model_prob, "ensemble_hard_vote_rate": vote_rate}

    return {
        "probabilities": prob_block,
        "prediction": yhat,
        "threshold_used": thr,
        "W_components": W.flatten().tolist()
    }

# セッション状態の初期化
if 'scores' not in st.session_state:
    st.session_state.scores = {}
    # 全項目を0で初期化
    mgc_items = ["MGC_ptosis", "MGC_diplopia", "MGC_eyelid_closure", "MGC_speech", 
                 "MGC_chewing", "MGC_swallowing", "MGC_respiration", "MGC_neck", 
                 "MGC_upper_limb", "MGC_lower_limb"]
    mgadl_items = ["MGADL_speech", "MGADL_chewing", "MGADL_swallowing", "MGADL_respiration",
                   "MGADL_toothbrushing", "MGADL_getting_up", "MGADL_diplopia", "MGADL_ptosis"]
    mgqol_items = [f"MGQOL{i}_" + name for i, name in enumerate([
        "dissatisfaction", "seeing", "eating", "social_activity_restriction",
        "hobby_entertainment", "family_role", "behavior_modification", "work_impact",
        "speaking", "driving", "feeling_down", "walking", "quick_action",
        "mental_crushing", "dressing"], 1)]
    
    for item in mgc_items + mgadl_items + mgqol_items:
        st.session_state.scores[item] = 0

def create_score_grid(item_key, item_name, max_score, current_value):
    """クリック可能なスコアグリッドを作成"""
    st.markdown(f"**{item_name}**")
    cols = st.columns(max_score + 1)
    
    for i in range(max_score + 1):
        with cols[i]:
            button_style = "primary" if current_value == i else "secondary"
            if st.button(str(i), key=f"{item_key}_{i}", type=button_style):
                st.session_state.scores[item_key] = i
                # MGADLからMGCへの自動マッピング
                if item_key.startswith("MGADL_"):
                    map_mgadl_to_mgc(item_key, i)
                st.rerun()

def map_mgadl_to_mgc(mgadl_key, value):
    """MGADLのスコアを対応するMGCに自動反映"""
    mapping = {
        "MGADL_speech": "MGC_speech",
        "MGADL_chewing": "MGC_chewing",
        "MGADL_swallowing": "MGC_swallowing",
        "MGADL_respiration": "MGC_respiration",
        "MGADL_diplopia": "MGC_diplopia",
        "MGADL_ptosis": "MGC_ptosis"
    }
    
    if mgadl_key in mapping:
        mgc_key = mapping[mgadl_key]
        # MGADLは0-3、MGCは0-5なので、比例変換
        mgc_value = min(5, int(value * 5 / 3))
        st.session_state.scores[mgc_key] = mgc_value

def display_input_section():
    """入力セクション"""
    st.markdown("<h2 style='text-align:center; color:#667eea;'>📝 Score Input</h2>", unsafe_allow_html=True)
    
    # MGADL入力（最初に）
    st.markdown("<div class='section-header'>1️⃣ MGADL (Activities of Daily Living) - 0-3 points</div>", 
                unsafe_allow_html=True)
    st.info("💡 MGADL scores automatically update corresponding MGC items")
    
    mgadl_items = [
        ("MGADL_speech", "Speech", 3),
        ("MGADL_chewing", "Chewing", 3),
        ("MGADL_swallowing", "Swallowing", 3),
        ("MGADL_respiration", "Breathing", 3),
        ("MGADL_toothbrushing", "Brushing/Combing", 3),
        ("MGADL_getting_up", "Rising from Chair", 3),
        ("MGADL_diplopia", "Double Vision", 3),
        ("MGADL_ptosis", "Eyelid Droop", 3)
    ]
    
    for key, name, max_score in mgadl_items:
        create_score_grid(key, name, max_score, st.session_state.scores[key])
    
    mgadl_total = sum([st.session_state.scores[key] for key, _, _ in mgadl_items])
    st.success(f"**MGADL Total: {mgadl_total}/24**")
    
    st.markdown("---")
    
    # MGC入力
    st.markdown("<div class='section-header'>2️⃣ MGC (Composite Score) - 0-5 points</div>", 
                unsafe_allow_html=True)
    st.info("🔄 Items marked with ※ are auto-filled from MGADL")
    
    mgc_items = [
        ("MGC_ptosis", "Ptosis ※", 5),
        ("MGC_diplopia", "Diplopia ※", 5),
        ("MGC_eyelid_closure", "Eyelid Closure", 5),
        ("MGC_speech", "Speech ※", 5),
        ("MGC_chewing", "Chewing ※", 5),
        ("MGC_swallowing", "Swallowing ※", 5),
        ("MGC_respiration", "Breathing ※", 5),
        ("MGC_neck", "Neck Flexion", 5),
        ("MGC_upper_limb", "Arm Raise", 5),
        ("MGC_lower_limb", "Leg Raise", 5)
    ]
    
    # 自動設定される項目
    auto_items = ["MGC_ptosis", "MGC_diplopia", "MGC_speech", "MGC_chewing", 
                  "MGC_swallowing", "MGC_respiration"]
    
    for key, name, max_score in mgc_items:
        if key in auto_items:
            st.markdown(f"**{name}**: {st.session_state.scores[key]} points (auto-filled from MGADL)")
        else:
            create_score_grid(key, name, max_score, st.session_state.scores[key])
    
    mgc_total = sum([st.session_state.scores[key] for key, _, _ in mgc_items])
    st.success(f"**MGC Total: {mgc_total}/50**")
    
    st.markdown("---")
    
    # MGQOL入力
    st.markdown("<div class='section-header'>3️⃣ MGQOL15 (Quality of Life) - 0-4 points</div>", 
                unsafe_allow_html=True)
    
    mgqol_items = [
        ("MGQOL1_dissatisfaction", "Frustration", 4),
        ("MGQOL2_seeing", "Ocular", 4),
        ("MGQOL3_eating", "Eating", 4),
        ("MGQOL4_social_activity_restriction", "Social", 4),
        ("MGQOL5_hobby_entertainment", "Hobbies", 4),
        ("MGQOL6_family_role", "Family", 4),
        ("MGQOL7_behavior_modification", "Planning", 4),
        ("MGQOL8_work_impact", "Occupation", 4),
        ("MGQOL9_speaking", "Speech", 4),
        ("MGQOL10_driving", "Driving", 4),
        ("MGQOL11_feeling_down", "Depression", 4),
        ("MGQOL12_walking", "Walking", 4),
        ("MGQOL13_quick_action", "Being Hurry", 4),
        ("MGQOL14_mental_crushing", "Overwhelmed", 4),
        ("MGQOL15_dressing", "Grooming", 4)
    ]
    
    for key, name, max_score in mgqol_items:
        create_score_grid(key, name, max_score, st.session_state.scores[key])
    
    mgqol_total = sum([st.session_state.scores[key] for key, _, _ in mgqol_items])
    st.warning(f"**MGQOL Total: {mgqol_total}/60**")
    
    return mgadl_total, mgc_total, mgqol_total

def display_diamond_chart(W_components, group_stats):
    """ダイヤモンドチャート表示（英語版）"""
    categories = ["module QOL", "module Diplopia", "module Ptosis", "module Systemic"]
    patient_values = W_components
    mm_values = [group_stats['MM_mean'][cat] for cat in categories]
    nonmm_values = [group_stats['nonMM_mean'][cat] for cat in categories]
    
    # 閉じたポリゴンにする
    categories_closed = categories + [categories[0]]
    patient_values_closed = patient_values + [patient_values[0]]
    mm_values_closed = mm_values + [mm_values[0]]
    nonmm_values_closed = nonmm_values + [nonmm_values[0]]
    
    fig = go.Figure()
    
    # 患者データ
    fig.add_trace(go.Scatterpolar(
        r=patient_values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(255, 99, 71, 0.3)',
        line=dict(color='#ff6347', width=3),
        name='Patient',
        mode='lines+markers',
        marker=dict(size=10, color='#ff6347')
    ))
    
    # MM群
    fig.add_trace(go.Scatterpolar(
        r=mm_values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(46, 204, 113, 0.2)',
        line=dict(color='#2ecc71', width=2),
        name='MM group',
        mode='lines+markers',
        marker=dict(size=8, color='#2ecc71')
    ))
    
    # non-MM群
    fig.add_trace(go.Scatterpolar(
        r=nonmm_values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.2)',
        line=dict(color='#3498db', width=2),
        name='non-MM group',
        mode='lines+markers',
        marker=dict(size=8, color='#3498db')
    ))
    
    max_value = max(max(patient_values), max(mm_values), max(nonmm_values))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_value * 1.2],
                tickfont=dict(size=12),
                gridcolor='lightgray'
            ),
            angularaxis=dict(
                tickfont=dict(size=13),
                rotation=90,
                direction='clockwise'
            )
        ),
        showlegend=True,
        height=400,
        margin=dict(l=20, r=20, t=30, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def display_results(result, group_stats, mgadl_total, mgc_total, mgqol_total):
    """結果表示"""
    st.markdown("---")
    st.markdown("<h2 style='text-align:center; color:#667eea;'>🔍 Prediction Results</h2>", 
                unsafe_allow_html=True)
    
    # 入力サマリー
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MGADL", f"{mgadl_total}/24", f"{mgadl_total/24*100:.0f}%")
    with col2:
        st.metric("MGC", f"{mgc_total}/50", f"{mgc_total/50*100:.0f}%")
    with col3:
        st.metric("MGQOL", f"{mgqol_total}/60", f"{mgqol_total/60*100:.0f}%")
    
    # アンサンブル予測結果
    ensemble_prob = result["probabilities"]["ensemble_soft"]
    prediction = result["prediction"]
    ensemble_pred = "MM or better" if prediction == 1 else "non-MM"
    
    color = "#2ecc71" if prediction == 1 else "#e74c3c"
    st.markdown(f"""
    <div style='
        background: {color};
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        margin: 1rem 0;
    '>
        <h1 style='margin:0;'>🎯 {ensemble_pred}</h1>
        <h2 style='margin:0.5rem 0;'>Probability: {ensemble_prob:.1%}</h2>
        <p style='margin:0;'>Threshold: {result["threshold_used"]:.2f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ダイヤモンドチャート
    st.markdown("### 💎 Group Comparison")
    fig = display_diamond_chart(result["W_components"], group_stats)
    st.plotly_chart(fig, use_container_width=True)
    
    # 詳細結果
    with st.expander("📊 Detailed Results"):
        # 各モデルの予測確率
        st.markdown("#### Model Predictions")
        model_probs = result["probabilities"]["per_model"]
        
        model_names_display = {
            'LR': 'Logistic Regression',
            'SVM': 'Support Vector Machine',
            'RF': 'Random Forest',
            'NB': 'Naive Bayes'
        }
        
        prob_df = pd.DataFrame({
            'Model': [model_names_display[k] for k in model_probs.keys()],
            'Probability': [f"{p:.1%}" for p in model_probs.values()],
            'Prediction': ['✅ MM' if p >= 0.5 else '❌ non-MM' for p in model_probs.values()]
        })
        st.dataframe(prob_df, hide_index=True)
        
        # NMF特徴量
        st.markdown("#### NMF Components")
        categories = ["module QOL", "module Diplopia", "module Ptosis", "module Systemic"]
        feature_df = pd.DataFrame({
            'Module': categories,
            'Patient': [f"{v:.4f}" for v in result["W_components"]],
            'MM Mean': [f"{group_stats['MM_mean'][cat]:.4f}" for cat in categories],
            'non-MM Mean': [f"{group_stats['nonMM_mean'][cat]:.4f}" for cat in categories]
        })
        st.dataframe(feature_df, hide_index=True)

def main():
    # ヘッダー
    st.markdown("""
    <div class='result-card'>
        <h1>🏥 MG MM Prediction System</h1>
        <p>Predicting treatment response from MG scores</p>
    </div>
    """, unsafe_allow_html=True)
    
    # モデル読み込み
    with st.spinner("Loading models..."):
        models = load_models()
    
    if models is None:
        st.error("Failed to load models")
        st.stop()
    
    # デフォルト列順（feature_cols.pklがない場合のフォールバック）
    DEFAULT_FEATURE_COLS = [
        "MGC_ptosis", "MGC_diplopia", "MGC_eyelid_closure", "MGC_speech", "MGC_chewing",
        "MGC_swallowing", "MGC_respiration", "MGC_neck", "MGC_upper_limb", "MGC_lower_limb",
        "MGADL_speech", "MGADL_chewing", "MGADL_swallowing", "MGADL_respiration",
        "MGADL_toothbrushing", "MGADL_getting_up", "MGADL_diplopia", "MGADL_ptosis",
        "MGQOL1_dissatisfaction", "MGQOL2_seeing", "MGQOL3_eating",
        "MGQOL4_social_activity_restriction", "MGQOL5_hobby_entertainment", "MGQOL6_family_role",
        "MGQOL7_behavior_modification", "MGQOL8_work_impact", "MGQOL9_speaking",
        "MGQOL10_driving", "MGQOL11_feeling_down", "MGQOL12_walking", "MGQOL13_quick_action",
        "MGQOL14_mental_crushing", "MGQOL15_dressing"
    ]
    
    FEATURE_COLS_EFF = models.get('feature_cols') or DEFAULT_FEATURE_COLS
    
    # 既存セッションの初期化を補強（不足キーだけ追加）
    if 'scores' not in st.session_state:
        st.session_state.scores = {key: 0 for key in FEATURE_COLS_EFF}
    else:
        for k in FEATURE_COLS_EFF:
            st.session_state.scores.setdefault(k, 0)
    
    # 入力セクション
    mgadl_total, mgc_total, mgqol_total = display_input_section()
    
    # ボタン配置
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("🔄 Reset", use_container_width=True):
            for key in st.session_state.scores:
                st.session_state.scores[key] = 0
            st.rerun()
    
    with col2:
        if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
            total_score = mgadl_total + mgc_total + mgqol_total
            if total_score == 0:
                st.warning("Please enter scores first")
            else:
                with st.spinner("Predicting..."):
                    try:
                        # predict_case関数を使用（学習時の列順と閾値を使用）
                        result = predict_case(
                            case_input=st.session_state.scores,
                            feature_cols=FEATURE_COLS_EFF,
                            scaler=models['scaler'],
                            nmf=models['nmf'],
                            models_cv=models['models_cv'],
                            thresholds=models.get('thresholds'),
                            ensemble_rule="soft"
                        )
                        
                        # 結果表示
                        display_results(result, models['group_stats'],
                                      mgadl_total, mgc_total, mgqol_total)
                    except ValueError as e:
                        st.error(f"Prediction error: {e}")
    
    # デバッグ情報
    with st.expander("🔧 Debug Info"):
        st.write("Current scores:")
        scores_df = pd.DataFrame([st.session_state.scores]).T
        scores_df.columns = ['Score']
        st.dataframe(scores_df)
        
        st.write("Model info:")
        st.write(f"- Feature columns: {len(FEATURE_COLS_EFF)} features")
        st.write(f"- Scaler range: {models['scaler'].feature_range}")
        if models.get('thresholds'):
            st.write(f"- Thresholds: {models['thresholds']}")

if __name__ == "__main__":
    main()