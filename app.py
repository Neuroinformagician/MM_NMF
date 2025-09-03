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
    layout="wide"
)

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

def create_input_form():
    """入力フォームを作成"""
    st.markdown("### 📝 患者データ入力")
    st.markdown("各項目のスコアを入力してください（0-5の範囲）")
    
    input_data = {}
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### MGC (Myasthenia Gravis Composite)")
        mgc_items = {
            "MGC_ptosis": "眼瞼下垂",
            "MGC_diplopia": "複視",
            "MGC_eyelid_closure": "眼瞼閉鎖",
            "MGC_speech": "構音障害",
            "MGC_chewing": "咀嚼",
            "MGC_swallowing": "嚥下",
            "MGC_respiration": "呼吸",
            "MGC_neck": "頸部屈曲/伸展",
            "MGC_upper_limb": "上肢",
            "MGC_lower_limb": "下肢"
        }
        
        for key, label in mgc_items.items():
            input_data[key] = st.slider(
                f"{label}",
                min_value=0,
                max_value=5,
                value=0,
                key=key
            )
    
    with col2:
        st.markdown("#### MGADL (MG Activities of Daily Living)")
        mgadl_items = {
            "MGADL_speech": "会話",
            "MGADL_chewing": "咀嚼",
            "MGADL_swallowing": "嚥下",
            "MGADL_respiration": "呼吸",
            "MGADL_toothbrushing": "歯磨き・櫛使用",
            "MGADL_getting_up": "椅子から立ち上がる",
            "MGADL_diplopia": "複視",
            "MGADL_ptosis": "眼瞼下垂"
        }
        
        for key, label in mgadl_items.items():
            input_data[key] = st.slider(
                f"{label}",
                min_value=0,
                max_value=3,
                value=0,
                key=key
            )
    
    with col3:
        st.markdown("#### MGQOL15 (MG Quality of Life)")
        mgqol_items = {
            "MGQOL1_dissatisfaction": "不満足感",
            "MGQOL2_seeing": "視覚",
            "MGQOL3_eating": "食事",
            "MGQOL4_social_activity_restriction": "社会活動制限",
            "MGQOL5_hobby_entertainment": "趣味・娯楽",
            "MGQOL6_family_role": "家族内役割",
            "MGQOL7_behavior_modification": "行動変更",
            "MGQOL8_work_impact": "仕事への影響"
        }
        
        for key, label in mgqol_items.items():
            input_data[key] = st.slider(
                f"{label}",
                min_value=0,
                max_value=4,
                value=0,
                key=key
            )
        
        mgqol_items2 = {
            "MGQOL9_speaking": "会話",
            "MGQOL10_driving": "運転",
            "MGQOL11_feeling_down": "気分の落ち込み",
            "MGQOL12_walking": "歩行",
            "MGQOL13_quick_action": "素早い動作",
            "MGQOL14_mental_crushing": "精神的圧迫感",
            "MGQOL15_dressing": "着替え"
        }
        
        for key, label in mgqol_items2.items():
            input_data[key] = st.slider(
                f"{label}",
                min_value=0,
                max_value=4,
                value=0,
                key=key
            )
    
    return input_data

def transform_to_nmf_features(input_data, models):
    """入力データをNMF特徴量に変換"""
    # DataFrameに変換
    df = pd.DataFrame([input_data])
    
    # 正規化
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    
    # NMF変換
    W = models['nmf'].transform(df_scaled)
    
    # モジュール名を付けてDataFrameに
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
        # 5つのfoldモデルで予測して平均を取る
        fold_probs = []
        for fold_model in models[model_type]:
            prob = fold_model.predict_proba(W_df)[0, 1]
            fold_probs.append(prob)
        
        avg_prob = np.mean(fold_probs)
        probabilities[display_name] = avg_prob
        predictions[display_name] = 1 if avg_prob >= 0.5 else 0
    
    return predictions, probabilities

def display_results(predictions, probabilities, W_df, group_stats):
    """結果を表示"""
    st.markdown("---")
    st.markdown("## 🔍 予測結果")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 モデル別予測確率")
        
        # アンサンブル予測（平均）
        ensemble_prob = np.mean(list(probabilities.values()))
        ensemble_pred = "MM or better" if ensemble_prob >= 0.5 else "non-MM"
        
        # メトリクス表示
        st.metric(
            label="アンサンブル予測",
            value=ensemble_pred,
            delta=f"確率: {ensemble_prob:.1%}"
        )
        
        # 各モデルの予測確率をバーチャートで表示
        prob_df = pd.DataFrame({
            'モデル': list(probabilities.keys()),
            '予測確率': list(probabilities.values())
        })
        
        fig = px.bar(
            prob_df, 
            x='予測確率', 
            y='モデル',
            orientation='h',
            color='予測確率',
            color_continuous_scale='RdYlGn',
            range_color=[0, 1]
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💎 群間比較（ダイヤモンドチャート）")
        
        # データ準備
        categories = W_df.columns.tolist()
        patient_values = W_df.values[0].tolist()
        
        # MM群とnon-MM群の平均値を取得
        mm_values = [group_stats['MM_mean'][cat] for cat in categories]
        nonmm_values = [group_stats['nonMM_mean'][cat] for cat in categories]
        
        # レーダーチャート（ダイヤモンド形）
        fig = go.Figure()
        
        # 患者データ
        fig.add_trace(go.Scatterpolar(
            r=patient_values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(color='red', width=2),
            name='患者データ'
        ))
        
        # MM群平均
        fig.add_trace(go.Scatterpolar(
            r=mm_values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(0, 255, 0, 0.1)',
            line=dict(color='green', width=2),
            name='MM群平均'
        ))
        
        # non-MM群平均
        fig.add_trace(go.Scatterpolar(
            r=nonmm_values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(0, 0, 255, 0.1)',
            line=dict(color='blue', width=2),
            name='non-MM群平均'
        ))
        
        # レイアウト設定（ダイヤモンド形にするため4軸）
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(max(patient_values), max(mm_values), max(nonmm_values)) * 1.2]
                )),
            showlegend=True,
            height=400,
            title="NMF特徴量の比較"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # NMF特徴量の詳細表示
    st.markdown("---")
    st.markdown("### 📊 NMF特徴量の詳細比較")
    
    # 数値表の作成
    comparison_df = pd.DataFrame({
        'モジュール': W_df.columns,
        '患者データ': W_df.values[0],
        'MM群平均': [group_stats['MM_mean'][cat] for cat in W_df.columns],
        'non-MM群平均': [group_stats['nonMM_mean'][cat] for cat in W_df.columns],
        'MM群との差': [abs(W_df.values[0][i] - group_stats['MM_mean'][cat]) for i, cat in enumerate(W_df.columns)],
        'non-MM群との差': [abs(W_df.values[0][i] - group_stats['nonMM_mean'][cat]) for i, cat in enumerate(W_df.columns)]
    })
    
    # より近い群を判定
    comparison_df['より近い群'] = comparison_df.apply(
        lambda row: 'MM群' if row['MM群との差'] < row['non-MM群との差'] else 'non-MM群', axis=1
    )
    
    # フォーマット調整
    comparison_df['患者データ'] = comparison_df['患者データ'].round(4)
    comparison_df['MM群平均'] = comparison_df['MM群平均'].round(4)
    comparison_df['non-MM群平均'] = comparison_df['non-MM群平均'].round(4)
    comparison_df['MM群との差'] = comparison_df['MM群との差'].round(4)
    comparison_df['non-MM群との差'] = comparison_df['non-MM群との差'].round(4)
    
    st.dataframe(comparison_df, hide_index=True)
    
    st.markdown("""
    📝 **解釈のポイント:**
    - **より近い群**: 患者データがどちらの群により類似しているかを示す
    - **差の値**: 小さいほど、その群との類似度が高い
    - MM群に近いほど治療反応性が良好であることを示唆
    """)
    
    # 詳細結果
    with st.expander("📋 詳細な予測結果"):
        result_df = pd.DataFrame({
            'モデル': list(predictions.keys()),
            '予測': ['MM or better' if p == 1 else 'non-MM' for p in predictions.values()],
            '確率': [f"{prob:.1%}" for prob in probabilities.values()]
        })
        st.dataframe(result_df, hide_index=True)
        
        # 判定基準の説明
        st.info(
            """
            **判定基準:**
            - 確率 ≥ 50%: MM or better（治療反応性良好）
            - 確率 < 50%: non-MM（治療反応性不良）
            
            **MM (Minimal Manifestations)**: 
            わずかな症状のみが残存している状態を指します。
            """
        )

def main():
    st.title("🏥 重症筋無力症（MG）MM判定システム")
    st.markdown("""
    このシステムは、MGC、MGADL、MGQOL15のスコアから、
    患者さんの治療反応性（MM or better）を予測します。
    """)
    
    # モデル読み込み
    with st.spinner("モデルを読み込んでいます..."):
        models = load_models()
    
    if models is None:
        st.error("モデルの読み込みに失敗しました。")
        st.stop()
    
    # サイドバーに説明を追加
    with st.sidebar:
        st.markdown("### 📖 使い方")
        st.markdown("""
        1. 各項目のスコアを入力
        2. 「予測を実行」ボタンをクリック
        3. 結果を確認
        """)
        
        st.markdown("### 🔢 スコアの範囲")
        st.markdown("""
        - **MGC**: 0-5点
        - **MGADL**: 0-3点
        - **MGQOL15**: 0-4点
        """)
        
        st.markdown("### ℹ️ About")
        st.markdown("""
        このシステムは、NMF（非負値行列因子分解）と
        機械学習を組み合わせて予測を行います。
        """)
    
    # 入力フォーム
    input_data = create_input_form()
    
    # 予測実行ボタン
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 予測を実行", type="primary", use_container_width=True):
            with st.spinner("予測中..."):
                # NMF特徴量に変換
                W_df = transform_to_nmf_features(input_data, models)
                
                # 予測
                predictions, probabilities = predict_mm(W_df, models)
                
                # 結果表示
                display_results(predictions, probabilities, W_df, models['group_stats'])
                
                # セッション状態に保存
                st.session_state['last_prediction'] = {
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'W_df': W_df,
                    'group_stats': models['group_stats']
                }
    
    # 前回の結果を表示
    if 'last_prediction' in st.session_state:
        if st.button("📊 前回の結果を再表示"):
            last = st.session_state['last_prediction']
            display_results(last['predictions'], last['probabilities'], last['W_df'], last['group_stats'])

if __name__ == "__main__":
    main()