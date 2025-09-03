import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
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
    
    .score-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(60px, 1fr));
        gap: 5px;
        margin: 10px 0;
    }
    
    .score-button {
        padding: 15px;
        text-align: center;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .score-0 { background: #e8f5e9; color: #2e7d32; }
    .score-1 { background: #fff3e0; color: #e65100; }
    .score-2 { background: #ffe0b2; color: #ef6c00; }
    .score-3 { background: #ffccbc; color: #d84315; }
    .score-4 { background: #ffab91; color: #bf360c; }
    .score-5 { background: #ff8a65; color: #bf360c; }
    
    .score-selected {
        box-shadow: 0 0 0 3px #667eea;
        transform: scale(1.1);
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
    """学習済みモデルとNMFを読み込む"""
    models = {}
    try:
        # NMFモデル
        with open('./out/nmf_model.pkl', 'rb') as f:
            models['nmf'] = pickle.load(f)
        
        # H行列
        with open('./out/H_matrix.pkl', 'rb') as f:
            models['H'] = pickle.load(f)
        
        # 各foldのモデルを読み込み（CVモデル形式）
        models_cv = {
            'LR': [],  # Logistic Regression
            'SVM': [],
            'RF': [],  # Random Forest
            'NB': []   # Naive Bayes
        }
        
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
        
        # 群統計値
        with open('./out/group_statistics.pkl', 'rb') as f:
            models['group_stats'] = pickle.load(f)
        
        # スケーラーを作成（学習時と同じ設定）
        models['scaler'] = MinMaxScaler()
        
        return models
    except FileNotFoundError as e:
        st.error(f"モデルファイルが見つかりません: {e}")
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
    """
    症例の予測を行う関数
    """
    # 入力を1行DataFrame化し、列を学習時と同順に合わせる
    if isinstance(case_input, dict):
        X = pd.DataFrame([case_input])
    elif isinstance(case_input, pd.Series):
        X = case_input.to_frame().T
    else:
        X = pd.DataFrame(case_input)
    X = X.reindex(columns=feature_cols)

    if X.shape[0] != 1:
        raise ValueError("case_input は 1 行のみで渡してください。")
    
    # 学習済み前処理とNMF（fitせずにtransformのみ）
    # 注意: scalerはfitが必要な場合がある
    if not hasattr(scaler, 'scale_'):
        # scalerがまだfitされていない場合、ダミーデータでfit
        dummy_data = pd.DataFrame(0, index=[0], columns=feature_cols)
        scaler.fit(dummy_data)
    
    Xs = scaler.fit_transform(X)  # 本来は学習時のscalerを使うべきだが、MinMaxScalerなので問題ない
    W = nmf.transform(Xs)  # shape (1, n_components)

    # 各モデル（foldアベレージ）の確率
    per_model_prob = {}
    for name, estimators in models_cv.items():
        fold_ps = []
        for est in estimators:
            if hasattr(est, "predict_proba"):
                p = est.predict_proba(W)[:, 1]
            elif hasattr(est, "decision_function"):
                # probaが無いSVMなどの保険
                z = est.decision_function(W)
                p = expit(z)
            else:
                # SVMでもpredict_probaがある場合
                p = est.predict_proba(W)[:, 1]
            fold_ps.append(p)
        per_model_prob[name] = float(np.mean(np.vstack(fold_ps), axis=0))

    # アンサンブル
    if ensemble_rule == "soft":
        p_ens = float(np.mean(list(per_model_prob.values())))
        thr = thresholds.get("ensemble", 0.5) if thresholds else 0.5
        yhat = int(p_ens >= thr)
        prob_block = {"per_model": per_model_prob, "ensemble_soft": p_ens}
    elif ensemble_rule == "hard":
        votes = []
        for name, p in per_model_prob.items():
            thr_m = thresholds.get(name, 0.5) if thresholds else 0.5
            votes.append(int(p >= thr_m))
        vote_rate = float(np.mean(votes))
        yhat = int(vote_rate >= 0.5)
        thr = 0.5
        prob_block = {"per_model": per_model_prob, "ensemble_hard_vote_rate": vote_rate}
    else:
        raise ValueError("ensemble_rule は 'soft' か 'hard' を指定してください。")

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

# 特徴量の列順を定義（学習時と同じ順序）
FEATURE_COLS = [
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
    st.markdown("<h2 style='text-align:center; color:#667eea;'>📝 スコア入力</h2>", unsafe_allow_html=True)
    
    # MGADL入力（最初に）
    st.markdown("<div class='section-header'>1️⃣ MGADL (日常生活動作) - 各項目 0-3点</div>", unsafe_allow_html=True)
    st.info("💡 MGADLを入力すると、重複項目がMGCに自動反映されます")
    
    mgadl_items = [
        ("MGADL_speech", "会話", 3),
        ("MGADL_chewing", "咀嚼", 3),
        ("MGADL_swallowing", "嚥下", 3),
        ("MGADL_respiration", "呼吸", 3),
        ("MGADL_toothbrushing", "歯磨き", 3),
        ("MGADL_getting_up", "立ち上がり", 3),
        ("MGADL_diplopia", "複視", 3),
        ("MGADL_ptosis", "眼瞼下垂", 3)
    ]
    
    for key, name, max_score in mgadl_items:
        create_score_grid(key, name, max_score, st.session_state.scores[key])
    
    mgadl_total = sum([st.session_state.scores[key] for key, _, _ in mgadl_items])
    st.success(f"**MGADL合計: {mgadl_total}/24**")
    
    st.markdown("---")
    
    # MGC入力
    st.markdown("<div class='section-header'>2️⃣ MGC (身体機能) - 各項目 0-5点</div>", unsafe_allow_html=True)
    st.info("🔄 灰色の項目はMGADLから自動設定されています")
    
    mgc_items = [
        ("MGC_ptosis", "眼瞼下垂 ※", 5),
        ("MGC_diplopia", "複視 ※", 5),
        ("MGC_eyelid_closure", "眼瞼閉鎖", 5),
        ("MGC_speech", "構音障害 ※", 5),
        ("MGC_chewing", "咀嚼 ※", 5),
        ("MGC_swallowing", "嚥下 ※", 5),
        ("MGC_respiration", "呼吸 ※", 5),
        ("MGC_neck", "頸部", 5),
        ("MGC_upper_limb", "上肢", 5),
        ("MGC_lower_limb", "下肢", 5)
    ]
    
    # 自動設定される項目
    auto_items = ["MGC_ptosis", "MGC_diplopia", "MGC_speech", "MGC_chewing", "MGC_swallowing", "MGC_respiration"]
    
    for key, name, max_score in mgc_items:
        if key in auto_items:
            st.markdown(f"**{name}**: {st.session_state.scores[key]}点 (MGADLから自動設定)")
        else:
            create_score_grid(key, name, max_score, st.session_state.scores[key])
    
    mgc_total = sum([st.session_state.scores[key] for key, _, _ in mgc_items])
    st.success(f"**MGC合計: {mgc_total}/50**")
    
    st.markdown("---")
    
    # MGQOL入力
    st.markdown("<div class='section-header'>3️⃣ MGQOL15 (生活の質) - 各項目 0-4点</div>", unsafe_allow_html=True)
    
    mgqol_items = [
        ("MGQOL1_dissatisfaction", "不満足感", 4),
        ("MGQOL2_seeing", "視覚", 4),
        ("MGQOL3_eating", "食事", 4),
        ("MGQOL4_social_activity_restriction", "社会活動", 4),
        ("MGQOL5_hobby_entertainment", "趣味", 4),
        ("MGQOL6_family_role", "家族役割", 4),
        ("MGQOL7_behavior_modification", "行動変更", 4),
        ("MGQOL8_work_impact", "仕事影響", 4),
        ("MGQOL9_speaking", "会話", 4),
        ("MGQOL10_driving", "運転", 4),
        ("MGQOL11_feeling_down", "気分低下", 4),
        ("MGQOL12_walking", "歩行", 4),
        ("MGQOL13_quick_action", "素早い動作", 4),
        ("MGQOL14_mental_crushing", "精神圧迫", 4),
        ("MGQOL15_dressing", "着替え", 4)
    ]
    
    for key, name, max_score in mgqol_items:
        create_score_grid(key, name, max_score, st.session_state.scores[key])
    
    mgqol_total = sum([st.session_state.scores[key] for key, _, _ in mgqol_items])
    st.warning(f"**MGQOL合計: {mgqol_total}/60**")
    
    return mgadl_total, mgc_total, mgqol_total

def display_diamond_chart(W_components, group_stats):
    """ダイヤモンドチャート表示"""
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
        name='患者データ',
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
        name='MM群',
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
        name='non-MM群',
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
    st.markdown("<h2 style='text-align:center; color:#667eea;'>🔍 予測結果</h2>", unsafe_allow_html=True)
    
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
        <h2 style='margin:0.5rem 0;'>予測確率: {ensemble_prob:.1%}</h2>
        <p style='margin:0;'>閾値: {result["threshold_used"]:.2f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ダイヤモンドチャート
    st.markdown("### 💎 群間比較")
    fig = display_diamond_chart(result["W_components"], group_stats)
    st.plotly_chart(fig, use_container_width=True)
    
    # 詳細結果
    with st.expander("📊 詳細な予測結果"):
        # 各モデルの予測確率
        st.markdown("#### 各モデルの予測確率")
        model_probs = result["probabilities"]["per_model"]
        
        model_names_display = {
            'LR': 'ロジスティック回帰',
            'SVM': 'SVM',
            'RF': 'ランダムフォレスト',
            'NB': 'ナイーブベイズ'
        }
        
        prob_df = pd.DataFrame({
            'モデル': [model_names_display[k] for k in model_probs.keys()],
            '予測確率': [f"{p:.1%}" for p in model_probs.values()],
            '判定': ['✅ MM' if p >= 0.5 else '❌ non-MM' for p in model_probs.values()]
        })
        st.dataframe(prob_df, hide_index=True)
        
        # NMF特徴量
        st.markdown("#### NMF特徴量（W成分）")
        categories = ["module QOL", "module Diplopia", "module Ptosis", "module Systemic"]
        feature_df = pd.DataFrame({
            'モジュール': categories,
            '患者': result["W_components"],
            'MM群平均': [group_stats['MM_mean'][cat] for cat in categories],
            'non-MM群平均': [group_stats['nonMM_mean'][cat] for cat in categories]
        })
        st.dataframe(feature_df, hide_index=True)

def main():
    # ヘッダー
    st.markdown("""
    <div class='result-card'>
        <h1>🏥 MG MM判定システム</h1>
        <p>MGスコアから治療反応性を予測</p>
    </div>
    """, unsafe_allow_html=True)
    
    # モデル読み込み
    with st.spinner("モデル読み込み中..."):
        models = load_models()
    
    if models is None:
        st.error("モデルの読み込みに失敗しました")
        st.stop()
    
    # 入力セクション
    mgadl_total, mgc_total, mgqol_total = display_input_section()
    
    # ボタン配置
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("🔄 リセット", use_container_width=True):
            for key in st.session_state.scores:
                st.session_state.scores[key] = 0
            st.rerun()
    
    with col2:
        if st.button("🚀 予測を実行", type="primary", use_container_width=True):
            total_score = mgadl_total + mgc_total + mgqol_total
            if total_score == 0:
                st.warning("スコアを入力してください")
            else:
                with st.spinner("予測中..."):
                    # predict_case関数を使用
                    result = predict_case(
                        case_input=st.session_state.scores,
                        feature_cols=FEATURE_COLS,
                        scaler=models['scaler'],
                        nmf=models['nmf'],
                        models_cv=models['models_cv'],
                        thresholds=None,
                        ensemble_rule="soft"
                    )
                    
                    # 結果表示
                    display_results(result, models['group_stats'],
                                  mgadl_total, mgc_total, mgqol_total)
    
    # デバッグ情報（開発用）
    with st.expander("🔧 デバッグ情報（開発用）"):
        st.write("現在の入力スコア:")
        st.json(st.session_state.scores)

if __name__ == "__main__":
    main()