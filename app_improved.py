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

# macOSらしいカスタムCSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .stTabs {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }
    
    .score-table {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        text-align: center;
    }
    
    .chart-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        color: white;
        font-weight: 600;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
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
        
        # 各foldのモデルを読み込み
        model_types = ['svm', 'logistic_regression', 'random_forest', 'naive_bayes']
        for model_type in model_types:
            models[model_type] = []
            for fold in range(5):
                with open(f'./out/{model_type}_fold_{fold}.pkl', 'rb') as f:
                    models[model_type].append(pickle.load(f))
        
        # 群統計値を読み込み
        with open('./out/group_statistics.pkl', 'rb') as f:
            models['group_stats'] = pickle.load(f)
        
        return models
    except FileNotFoundError as e:
        st.error(f"モデルファイルが見つかりません: {e}")
        return None

def create_score_table(items, score_ranges, current_scores, key_prefix):
    """スコア入力テーブルを作成"""
    scores = {}
    
    for item_key, item_name in items.items():
        st.markdown(f"**{item_name}**")
        
        # スコア選択のためのラジオボタン
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        options = []
        descriptions = []
        
        if "MGC" in item_key:
            options = ["0", "1", "2", "3", "4", "5"]
            descriptions = ["正常", "軽度", "中等度", "重度", "高度", "最重度"]
        elif "MGADL" in item_key:
            options = ["0", "1", "2", "3"]
            descriptions = ["正常", "軽度異常", "中等度異常", "重度異常"]
        else:  # MGQOL
            options = ["0", "1", "2", "3", "4"]
            descriptions = ["全くない", "少しある", "ある程度", "かなり", "非常に多い"]
        
        # ラジオボタンを横並びで配置
        cols = st.columns(len(options))
        selected_score = current_scores.get(item_key, 0)
        
        for i, (option, desc) in enumerate(zip(options, descriptions)):
            with cols[i]:
                if st.button(f"{option}\n{desc}", key=f"{key_prefix}_{item_key}_{i}"):
                    selected_score = int(option)
                    st.session_state[f"score_{item_key}"] = selected_score
        
        scores[item_key] = st.session_state.get(f"score_{item_key}", 0)
        
        # 現在の選択を表示
        st.markdown(f"<div style='text-align: center; color: #667eea; font-weight: bold;'>選択中: {scores[item_key]}点</div>", 
                   unsafe_allow_html=True)
        st.markdown("---")
    
    return scores

def create_input_tabs():
    """タブベースの入力フォームを作成"""
    st.markdown("<h2 style='text-align: center; color: #667eea; margin-bottom: 2rem;'>📝 患者データ入力</h2>", 
                unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔵 MGC (身体機能)", "🟢 MGADL (日常生活)", "🟡 MGQOL15 (生活の質)"])
    
    all_scores = {}
    
    with tab1:
        st.markdown("### MGC (Myasthenia Gravis Composite)")
        st.markdown("身体機能の評価を行います（各項目0-5点）")
        
        mgc_items = {
            "MGC_ptosis": "眼瞼下垂 (Ptosis)",
            "MGC_diplopia": "複視 (Diplopia)", 
            "MGC_eyelid_closure": "眼瞼閉鎖 (Eyelid Closure)",
            "MGC_speech": "構音障害 (Speech)",
            "MGC_chewing": "咀嚼 (Chewing)",
            "MGC_swallowing": "嚥下 (Swallowing)",
            "MGC_respiration": "呼吸 (Respiration)",
            "MGC_neck": "頸部屈曲/伸展 (Neck)",
            "MGC_upper_limb": "上肢 (Upper Limb)",
            "MGC_lower_limb": "下肢 (Lower Limb)"
        }
        
        mgc_scores = create_score_table(mgc_items, (0, 5), {}, "mgc")
        all_scores.update(mgc_scores)
        
        # MGC合計スコア表示
        mgc_total = sum(mgc_scores.values())
        st.markdown(f"""
        <div class="metric-card">
            <h3>MGC 合計スコア</h3>
            <h1>{mgc_total}/50</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### MGADL (MG Activities of Daily Living)")
        st.markdown("日常生活動作の評価を行います（各項目0-3点）")
        
        mgadl_items = {
            "MGADL_speech": "会話 (Speech)",
            "MGADL_chewing": "咀嚼 (Chewing)",
            "MGADL_swallowing": "嚥下 (Swallowing)",
            "MGADL_respiration": "呼吸 (Respiration)",
            "MGADL_toothbrushing": "歯磨き・櫛使用 (Toothbrushing/Combing)",
            "MGADL_getting_up": "椅子から立ち上がる (Getting Up from Chair)",
            "MGADL_diplopia": "複視 (Diplopia)",
            "MGADL_ptosis": "眼瞼下垂 (Ptosis)"
        }
        
        mgadl_scores = create_score_table(mgadl_items, (0, 3), {}, "mgadl")
        all_scores.update(mgadl_scores)
        
        # MGADL合計スコア表示
        mgadl_total = sum(mgadl_scores.values())
        st.markdown(f"""
        <div class="metric-card">
            <h3>MGADL 合計スコア</h3>
            <h1>{mgadl_total}/24</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### MGQOL15 (MG Quality of Life)")
        st.markdown("生活の質の評価を行います（各項目0-4点）")
        
        mgqol_items = {
            "MGQOL1_dissatisfaction": "不満足感",
            "MGQOL2_seeing": "視覚機能",
            "MGQOL3_eating": "食事",
            "MGQOL4_social_activity_restriction": "社会活動制限",
            "MGQOL5_hobby_entertainment": "趣味・娯楽",
            "MGQOL6_family_role": "家族内役割",
            "MGQOL7_behavior_modification": "行動変更",
            "MGQOL8_work_impact": "仕事への影響",
            "MGQOL9_speaking": "会話",
            "MGQOL10_driving": "運転",
            "MGQOL11_feeling_down": "気分の落ち込み",
            "MGQOL12_walking": "歩行",
            "MGQOL13_quick_action": "素早い動作",
            "MGQOL14_mental_crushing": "精神的圧迫感",
            "MGQOL15_dressing": "着替え"
        }
        
        mgqol_scores = create_score_table(mgqol_items, (0, 4), {}, "mgqol")
        all_scores.update(mgqol_scores)
        
        # MGQOL合計スコア表示
        mgqol_total = sum(mgqol_scores.values())
        st.markdown(f"""
        <div class="metric-card">
            <h3>MGQOL15 合計スコア</h3>
            <h1>{mgqol_total}/60</h1>
        </div>
        """, unsafe_allow_html=True)
    
    return all_scores

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

def display_large_diamond_chart(W_df, group_stats):
    """大きなダイヤモンドチャートを表示"""
    st.markdown("""
    <div class="chart-container">
        <h2 style='text-align: center; color: #667eea; margin-bottom: 2rem;'>
            💎 群間比較（ダイヤモンドチャート）
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # データ準備
    categories = W_df.columns.tolist()
    patient_values = W_df.values[0].tolist()
    mm_values = [group_stats['MM_mean'][cat] for cat in categories]
    nonmm_values = [group_stats['nonMM_mean'][cat] for cat in categories]
    
    # 大きなレーダーチャート
    fig = go.Figure()
    
    # 患者データ
    fig.add_trace(go.Scatterpolar(
        r=patient_values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(255, 99, 71, 0.3)',
        line=dict(color='#ff6347', width=4),
        name='あなたの患者データ',
        hovertemplate='%{theta}<br>値: %{r:.4f}<extra></extra>'
    ))
    
    # MM群平均
    fig.add_trace(go.Scatterpolar(
        r=mm_values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(46, 204, 113, 0.2)',
        line=dict(color='#2ecc71', width=3),
        name='MM群平均 (治療反応良好)',
        hovertemplate='%{theta}<br>MM群平均: %{r:.4f}<extra></extra>'
    ))
    
    # non-MM群平均
    fig.add_trace(go.Scatterpolar(
        r=nonmm_values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.2)',
        line=dict(color='#3498db', width=3),
        name='non-MM群平均 (治療反応不良)',
        hovertemplate='%{theta}<br>non-MM群平均: %{r:.4f}<extra></extra>'
    ))
    
    # レイアウト設定
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(patient_values), max(mm_values), max(nonmm_values)) * 1.3],
                tickfont=dict(size=14),
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            angularaxis=dict(
                tickfont=dict(size=16, color='#2c3e50'),
                rotation=90
            ),
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        showlegend=True,
        height=600,
        width=600,
        font=dict(size=14, color='#2c3e50'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(128, 128, 128, 0.2)',
            borderwidth=1
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_results(predictions, probabilities, W_df, group_stats):
    """改善された結果表示"""
    st.markdown("---")
    st.markdown("<h1 style='text-align: center; color: #667eea; margin: 2rem 0;'>🔍 予測結果</h1>", 
                unsafe_allow_html=True)
    
    # アンサンブル予測
    ensemble_prob = np.mean(list(probabilities.values()))
    ensemble_pred = "MM or better" if ensemble_prob >= 0.5 else "non-MM"
    
    # メイン結果カード
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        color = "#2ecc71" if ensemble_prob >= 0.5 else "#e74c3c"
        st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, {color} 0%, {color}aa 100%);
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
            margin: 2rem 0;
        '>
            <h2>🎯 最終予測結果</h2>
            <h1 style='font-size: 3rem; margin: 1rem 0;'>{ensemble_pred}</h1>
            <h3>予測確率: {ensemble_prob:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # 大きなダイヤモンドチャート
    display_large_diamond_chart(W_df, group_stats)
    
    # モデル別予測確率
    st.markdown("### 📊 各機械学習モデルの予測詳細")
    
    prob_df = pd.DataFrame({
        'モデル': list(probabilities.keys()),
        '予測確率': list(probabilities.values()),
        '判定': ['MM or better' if p >= 0.5 else 'non-MM' for p in probabilities.values()]
    })
    
    # 横棒グラフ
    fig_bar = px.bar(
        prob_df,
        x='予測確率',
        y='モデル',
        orientation='h',
        color='予測確率',
        color_continuous_scale=['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71'],
        range_color=[0, 1],
        text='判定'
    )
    fig_bar.add_vline(x=0.5, line_dash="dash", line_color="gray", line_width=2)
    fig_bar.update_layout(
        height=400,
        title="各モデルの予測確率",
        xaxis_title="予測確率",
        yaxis_title="機械学習モデル",
        font=dict(size=14)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

def main():
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 20px; margin-bottom: 2rem; color: white; box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);'>
        <h1 style='font-size: 3rem; margin-bottom: 1rem;'>🏥 MG MM判定システム</h1>
        <p style='font-size: 1.2rem; opacity: 0.9;'>
            重症筋無力症患者の治療反応性を高精度で予測します<br>
            MGC・MGADL・MGQOL15スコアから機械学習による判定を行います
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # モデル読み込み
    with st.spinner("🔄 モデルを読み込んでいます..."):
        models = load_models()
    
    if models is None:
        st.error("❌ モデルの読み込みに失敗しました。")
        st.stop()
    
    # 入力フォーム
    input_data = create_input_tabs()
    
    # 予測実行ボタン
    st.markdown("<div style='text-align: center; margin: 3rem 0;'>", unsafe_allow_html=True)
    if st.button("🚀 予測を実行", type="primary"):
        if sum(input_data.values()) == 0:
            st.warning("⚠️ スコアを入力してから予測を実行してください。")
        else:
            with st.spinner("🔍 予測を実行中..."):
                # NMF特徴量に変換
                W_df = transform_to_nmf_features(input_data, models)
                
                # 予測
                predictions, probabilities = predict_mm(W_df, models)
                
                # 結果表示
                display_results(predictions, probabilities, W_df, models['group_stats'])
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()