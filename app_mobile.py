import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="MG MM判定システム",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# モバイル最適化CSS
st.markdown("""
<style>
    /* 余白を最小化 */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }
    
    .main {
        padding: 0;
    }
    
    /* タブスタイル */
    .stTabs {
        background: white;
        border-radius: 10px;
        padding: 0.5rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.2rem;
        background: #f0f2f6;
        border-radius: 8px;
        padding: 0.2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 0.9rem;
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
    }
    
    .stTabs [aria-selected="true"] {
        background: #667eea;
        color: white;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* ボタングリッド */
    .score-button {
        width: 100%;
        padding: 0.4rem;
        margin: 0.1rem 0;
        border-radius: 8px;
        font-size: 0.85rem;
    }
    
    /* 結果カード */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1rem;
        color: white;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin: 0.5rem 0;
    }
    
    /* スマホ対応 */
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab"] {
            font-size: 0.8rem;
            padding: 0.2rem 0.4rem;
        }
        
        div[data-testid="column"] {
            padding: 0 0.2rem;
        }
    }
    
    /* ボタンを小さく */
    .stButton > button {
        padding: 0.3rem 1rem;
        font-size: 0.9rem;
        border-radius: 8px;
    }
    
    /* セクションの間隔を詰める */
    .element-container {
        margin: 0.3rem 0;
    }
    
    h1 {
        font-size: 1.5rem;
        margin: 0.5rem 0;
    }
    
    h2 {
        font-size: 1.3rem;
        margin: 0.4rem 0;
    }
    
    h3 {
        font-size: 1.1rem;
        margin: 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """学習済みモデルとNMFを読み込む"""
    models = {}
    try:
        with open('./out/nmf_model.pkl', 'rb') as f:
            models['nmf'] = pickle.load(f)
        
        with open('./out/H_matrix.pkl', 'rb') as f:
            models['H'] = pickle.load(f)
        
        model_types = ['svm', 'logistic_regression', 'random_forest', 'naive_bayes']
        for model_type in model_types:
            models[model_type] = []
            for fold in range(5):
                with open(f'./out/{model_type}_fold_{fold}.pkl', 'rb') as f:
                    models[model_type].append(pickle.load(f))
        
        with open('./out/group_statistics.pkl', 'rb') as f:
            models['group_stats'] = pickle.load(f)
        
        return models
    except FileNotFoundError as e:
        st.error(f"モデルファイルが見つかりません: {e}")
        return None

def create_compact_score_input():
    """コンパクトなスコア入力フォーム"""
    
    # セッション状態の初期化
    if 'scores' not in st.session_state:
        st.session_state.scores = {}
    
    st.markdown("<h3 style='text-align:center; color:#667eea; margin:0.5rem 0;'>📝 患者データ入力</h3>", unsafe_allow_html=True)
    
    # タブで分割
    tab1, tab2, tab3 = st.tabs(["MGC", "MGADL", "MGQOL"])
    
    # MGC入力
    with tab1:
        mgc_items = [
            ("MGC_ptosis", "眼瞼下垂", 5),
            ("MGC_diplopia", "複視", 5),
            ("MGC_eyelid_closure", "眼瞼閉鎖", 5),
            ("MGC_speech", "構音障害", 5),
            ("MGC_chewing", "咀嚼", 5),
            ("MGC_swallowing", "嚥下", 5),
            ("MGC_respiration", "呼吸", 5),
            ("MGC_neck", "頸部", 5),
            ("MGC_upper_limb", "上肢", 5),
            ("MGC_lower_limb", "下肢", 5)
        ]
        
        # 2列レイアウト
        for i in range(0, len(mgc_items), 2):
            col1, col2 = st.columns(2)
            
            with col1:
                if i < len(mgc_items):
                    key, label, max_val = mgc_items[i]
                    st.session_state.scores[key] = st.selectbox(
                        label,
                        options=list(range(max_val + 1)),
                        index=st.session_state.scores.get(key, 0),
                        key=f"select_{key}"
                    )
            
            with col2:
                if i + 1 < len(mgc_items):
                    key, label, max_val = mgc_items[i + 1]
                    st.session_state.scores[key] = st.selectbox(
                        label,
                        options=list(range(max_val + 1)),
                        index=st.session_state.scores.get(key, 0),
                        key=f"select_{key}"
                    )
        
        mgc_total = sum([st.session_state.scores.get(item[0], 0) for item in mgc_items])
        st.info(f"**MGC合計: {mgc_total}/50**")
    
    # MGADL入力
    with tab2:
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
        
        for i in range(0, len(mgadl_items), 2):
            col1, col2 = st.columns(2)
            
            with col1:
                if i < len(mgadl_items):
                    key, label, max_val = mgadl_items[i]
                    st.session_state.scores[key] = st.selectbox(
                        label,
                        options=list(range(max_val + 1)),
                        index=st.session_state.scores.get(key, 0),
                        key=f"select_{key}"
                    )
            
            with col2:
                if i + 1 < len(mgadl_items):
                    key, label, max_val = mgadl_items[i + 1]
                    st.session_state.scores[key] = st.selectbox(
                        label,
                        options=list(range(max_val + 1)),
                        index=st.session_state.scores.get(key, 0),
                        key=f"select_{key}"
                    )
        
        mgadl_total = sum([st.session_state.scores.get(item[0], 0) for item in mgadl_items])
        st.success(f"**MGADL合計: {mgadl_total}/24**")
    
    # MGQOL入力
    with tab3:
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
        
        # 3列レイアウトでコンパクトに
        for i in range(0, len(mgqol_items), 3):
            cols = st.columns(3)
            
            for j in range(3):
                if i + j < len(mgqol_items):
                    with cols[j]:
                        key, label, max_val = mgqol_items[i + j]
                        st.session_state.scores[key] = st.selectbox(
                            label,
                            options=list(range(max_val + 1)),
                            index=st.session_state.scores.get(key, 0),
                            key=f"select_{key}"
                        )
        
        mgqol_total = sum([st.session_state.scores.get(item[0], 0) for item in mgqol_items])
        st.warning(f"**MGQOL合計: {mgqol_total}/60**")
    
    return st.session_state.scores

def display_input_summary(scores):
    """入力値のサマリー表示"""
    mgc_keys = [k for k in scores.keys() if k.startswith('MGC_')]
    mgadl_keys = [k for k in scores.keys() if k.startswith('MGADL_')]
    mgqol_keys = [k for k in scores.keys() if k.startswith('MGQOL')]
    
    mgc_total = sum([scores.get(k, 0) for k in mgc_keys])
    mgadl_total = sum([scores.get(k, 0) for k in mgadl_keys])
    mgqol_total = sum([scores.get(k, 0) for k in mgqol_keys])
    
    st.markdown("### 📊 入力スコアサマリー")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("MGC", f"{mgc_total}/50", f"{mgc_total/50*100:.0f}%")
    with col2:
        st.metric("MGADL", f"{mgadl_total}/24", f"{mgadl_total/24*100:.0f}%")
    with col3:
        st.metric("MGQOL", f"{mgqol_total}/60", f"{mgqol_total/60*100:.0f}%")

def transform_to_nmf_features(input_data, models):
    """入力データをNMF特徴量に変換"""
    df = pd.DataFrame([input_data])
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    W = models['nmf'].transform(df_scaled)
    
    feature_names = ["module QOL", "module Diplopia", "module Ptosis", "module Systemic"]
    W_df = pd.DataFrame(W, columns=feature_names)
    
    return W_df

def predict_mm(W_df, models):
    """MM判定を行う"""
    predictions = {}
    probabilities = {}
    
    model_names = {
        'svm': 'SVM',
        'logistic_regression': 'ロジスティック回帰',
        'random_forest': 'ランダムフォレスト',
        'naive_bayes': 'ナイーブベイズ'
    }
    
    for model_type, display_name in model_names.items():
        fold_probs = []
        for fold_model in models[model_type]:
            prob = fold_model.predict_proba(W_df)[0, 1]
            fold_probs.append(prob)
        
        avg_prob = np.mean(fold_probs)
        probabilities[display_name] = avg_prob
        predictions[display_name] = 1 if avg_prob >= 0.5 else 0
    
    return predictions, probabilities

def display_compact_diamond_chart(W_df, group_stats):
    """改良されたダイヤモンドチャート（線が消えない）"""
    categories = W_df.columns.tolist()
    patient_values = W_df.values[0].tolist()
    mm_values = [group_stats['MM_mean'][cat] for cat in categories]
    nonmm_values = [group_stats['nonMM_mean'][cat] for cat in categories]
    
    # 閉じたポリゴンにするため最初の値を末尾に追加
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
        marker=dict(size=8)
    ))
    
    # MM群
    fig.add_trace(go.Scatterpolar(
        r=mm_values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(46, 204, 113, 0.2)',
        line=dict(color='#2ecc71', width=2.5),
        name='MM群',
        mode='lines+markers',
        marker=dict(size=6)
    ))
    
    # non-MM群
    fig.add_trace(go.Scatterpolar(
        r=nonmm_values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.2)',
        line=dict(color='#3498db', width=2.5),
        name='non-MM群',
        mode='lines+markers',
        marker=dict(size=6)
    ))
    
    # レイアウト最適化
    max_value = max(max(patient_values), max(mm_values), max(nonmm_values))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_value * 1.2],
                tickfont=dict(size=12),
                gridcolor='lightgray',
                gridwidth=1
            ),
            angularaxis=dict(
                tickfont=dict(size=13, color='#2c3e50'),
                rotation=90,
                direction='clockwise',
                gridcolor='lightgray',
                gridwidth=1
            )
        ),
        showlegend=True,
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig

def display_results(predictions, probabilities, W_df, group_stats, input_scores):
    """結果表示"""
    st.markdown("---")
    
    # 入力サマリー
    display_input_summary(input_scores)
    
    st.markdown("---")
    
    # アンサンブル予測
    ensemble_prob = np.mean(list(probabilities.values()))
    ensemble_pred = "MM or better" if ensemble_prob >= 0.5 else "non-MM"
    
    # メイン結果
    color = "#2ecc71" if ensemble_prob >= 0.5 else "#e74c3c"
    st.markdown(f"""
    <div style='
        background: {color};
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        color: white;
        margin: 1rem 0;
    '>
        <h2 style='margin:0;'>🎯 予測結果</h2>
        <h1 style='margin:0.5rem 0;'>{ensemble_pred}</h1>
        <h3 style='margin:0;'>確率: {ensemble_prob:.1%}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # ダイヤモンドチャート
    st.markdown("### 💎 群間比較")
    fig = display_compact_diamond_chart(W_df, group_stats)
    st.plotly_chart(fig, use_container_width=True)
    
    # 各モデルの結果
    with st.expander("📊 詳細な予測結果"):
        # 予測確率の表
        prob_df = pd.DataFrame({
            'モデル': list(probabilities.keys()),
            '予測確率': [f"{p:.1%}" for p in probabilities.values()],
            '判定': ['✅ MM' if p >= 0.5 else '❌ non-MM' for p in probabilities.values()]
        })
        st.dataframe(prob_df, hide_index=True, use_container_width=True)
        
        # NMF特徴量
        st.markdown("#### NMF特徴量")
        feature_df = pd.DataFrame({
            'モジュール': W_df.columns,
            '患者': W_df.values[0].round(4),
            'MM群平均': [group_stats['MM_mean'][cat] for cat in W_df.columns],
            'non-MM群平均': [group_stats['nonMM_mean'][cat] for cat in W_df.columns]
        })
        st.dataframe(feature_df, hide_index=True, use_container_width=True)

def main():
    # コンパクトなヘッダー
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea, #764ba2); 
                border-radius: 10px; padding: 1rem; color: white; margin-bottom: 1rem;'>
        <h1 style='margin:0; text-align:center; font-size:1.8rem;'>🏥 MG MM判定システム</h1>
        <p style='margin:0.3rem 0 0 0; text-align:center; font-size:0.9rem;'>
            MGスコアから治療反応性を予測
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # モデル読み込み
    with st.spinner("モデル読み込み中..."):
        models = load_models()
    
    if models is None:
        st.error("モデルの読み込みに失敗しました")
        st.stop()
    
    # 入力フォーム
    input_scores = create_compact_score_input()
    
    # 予測実行ボタン（中央配置）
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 予測を実行", type="primary", use_container_width=True):
            if sum(input_scores.values()) == 0:
                st.warning("スコアを入力してください")
            else:
                with st.spinner("予測中..."):
                    # NMF特徴量に変換
                    W_df = transform_to_nmf_features(input_scores, models)
                    
                    # 予測
                    predictions, probabilities = predict_mm(W_df, models)
                    
                    # 結果表示
                    display_results(predictions, probabilities, W_df, models['group_stats'], input_scores)
    
    # リセットボタン
    if st.button("🔄 リセット", use_container_width=True):
        st.session_state.scores = {}
        st.rerun()

if __name__ == "__main__":
    main()