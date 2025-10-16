# MG予測アプリケーション 仕様書

## 1. アプリケーション概要

重症筋無力症（MG）患者の臨床スコアから、MM（Minimal Manifestations）以上の状態を予測するStreamlitアプリケーション。

### 目的
- 臨床現場での迅速な予測支援
- 3つの評価スケールの統合入力
- 視覚的な結果表示（レーダーチャート）

---

## 2. 機能要件

### 2.1 入力機能
- **MG-ADL** (8項目、各0-3点、合計24点)
- **MG Composite** (10項目、可変点数、合計50点)
- **MGQOL-15r** (15項目、各0-2点、合計30点)

### 2.2 データ変換
- 入力33項目を内部的に統一形式に変換
- MG-ADL → MG Composite の重複項目自動反映

### 2.3 予測機能
- NMFによる4モジュールスコア計算
  - module QOL
  - module Diplopia
  - module Ptosis
  - module Systemic
- 4つの機械学習モデルによるアンサンブル予測
  - SVM
  - Logistic Regression
  - Random Forest
  - Naive Bayes

### 2.4 結果表示
- **スコア表示**
  - 各スケールの合計点
  - 4つのモジュールスコア
- **予測結果**
  - 各モデルの予測確率
  - アンサンブル予測確率
  - MM/non-MM分類結果
- **レーダーチャート**
  - 患者のモジュールスコア
  - MM群平均
  - non-MM群平均
  - 3者の比較可視化

---

## 3. 画面構成

### 3.1 レイアウト

```
┌─────────────────────────────────────────────────┐
│ 🏥 MG予測システム                               │
├──────────────┬──────────────────────────────────┤
│ サイドバー   │ メインエリア                     │
│              │                                  │
│ 📋 MG-ADL    │ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ ├─ 項目1-8   │ 📊 入力サマリー                  │
│ │            │ ・MG-ADL: X/24点                 │
│ │            │ ・MG Composite: Y/50点           │
│ 📊 MG-C      │ ・MGQOL-15r: Z/30点              │
│ ├─ 項目1-10  │                                  │
│ │            │ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ │            │ 🎯 予測結果                      │
│ 💭 MGQOL     │ ・モジュールスコア表示           │
│ ├─ 項目1-15  │ ・各モデル予測確率               │
│ │            │ ・アンサンブル結果               │
│ │            │                                  │
│ [🔮 予測実行]│ 📈 レーダーチャート              │
│ [🔄 リセット]│ ・患者 vs MM群 vs non-MM群       │
│              │                                  │
└──────────────┴──────────────────────────────────┘
```

### 3.2 入力UI詳細

#### MG-ADL (サイドバー折りたたみ可能)
```python
st.sidebar.expander("📋 MG-ADL (8項目)", expanded=True)
  - 会話: selectbox [0-3]
  - 咀嚼: selectbox [0-3]
  - 嚥下: selectbox [0-3]
  - 呼吸: selectbox [0-3]
  - 歯磨き・櫛使用: selectbox [0-3]
  - 椅子からの立ち上がり: selectbox [0-3]
  - 複視: selectbox [0-3]
  - 眼瞼下垂: selectbox [0-3]
  合計: X/24点 (自動計算)
```

#### MG Composite (サイドバー折りたたみ可能)
```python
st.sidebar.expander("📊 MG Composite (10項目)", expanded=False)
  - 眼瞼下垂: selectbox [0,1,2,3点]
  - 複視: selectbox [0,1,3,4点]
  - 閉眼筋力: selectbox [0,0,1,2点]
  - 会話: selectbox [0,2,4,6点] ← ADLから自動反映
  - 咀嚼: selectbox [0,2,4,6点] ← ADLから自動反映
  - 嚥下: selectbox [0,2,5,6点] ← ADLから自動反映
  - 呼吸: selectbox [0,2,4,9点] ← ADLから自動反映
  - 頸部筋力: selectbox [0,1,3,4点]
  - 上肢筋力: selectbox [0,2,4,5点]
  - 下肢筋力: selectbox [0,2,4,5点]
  合計: Y/50点 (自動計算)
```

#### MGQOL-15r (サイドバー折りたたみ可能)
```python
st.sidebar.expander("💭 MGQOL-15r (15項目)", expanded=False)
  - Q1-Q15: selectbox [0,1,2]
  合計: Z/30点 (自動計算)
```

---

## 4. データフロー

```
[ユーザー入力]
    │
    ├─ MG-ADL (8項目) ────┐
    ├─ MG-C (10項目)      ├─→ [33項目統合]
    └─ MGQOL (15項目) ────┘         │
                                     ↓
                            [MinMaxScaler変換]
                                     │
                                     ↓
                              [NMF変換 (H行列)]
                                     │
                                     ↓
                          [4モジュールスコア算出]
                          - QOL
                          - Diplopia
                          - Ptosis
                          - Systemic
                                     │
                                     ↓
                    ┌────────────────┴────────────────┐
                    │                                  │
              [5-fold models]                   [レーダーチャート]
                    │                                  │
            ┌───────┼───────┐                          │
            │       │       │                          │
          SVM    LogReg   RF   NB                      │
            │       │       │                          │
            └───────┴───────┘                          │
                    │                                  │
              [アンサンブル平均]                       │
                    │                                  │
                    └──────────────┬───────────────────┘
                                   ↓
                            [結果表示画面]
```

---

## 5. 技術スタック

### 5.1 フレームワーク
- **Streamlit**: Webアプリフレームワーク
- **Python 3.8+**

### 5.2 ライブラリ
```python
streamlit >= 1.28.0
numpy
pandas
scikit-learn
matplotlib
plotly  # レーダーチャート用（インタラクティブ）
pickle  # モデル読み込み
```

### 5.3 既存資産の活用
- `./out/H_matrix.pkl`: NMF変換用H行列
- `./out/W_MM.pkl`: MM群/non-MM群の平均スコア
- `./out/module_names_reordered.pkl`: モジュール名順序
- `./out/{model}_fold_{i}.pkl`: 学習済みモデル (i=0-4)
- `./data/df_4th.csv`: スケーラー学習用データ

---

## 6. ファイル構成

```
MM_NMF_git/
├── streamlit_app.py           # メインアプリケーション
├── streamlit_config.py        # 設定ファイル（項目定義等）
├── streamlit_predictor.py     # 予測ロジック
├── streamlit_visualizer.py    # 可視化ロジック
├── requirements_streamlit.txt # 依存ライブラリ
├── data/
│   └── df_4th.csv
├── out/
│   ├── H_matrix.pkl
│   ├── W_MM.pkl
│   ├── module_names_reordered.pkl
│   ├── svm_fold_*.pkl
│   ├── logistic_regression_fold_*.pkl
│   ├── random_forest_fold_*.pkl
│   └── naive_bayes_fold_*.pkl
└── fig/
    └── (streamlitで生成したチャート保存用)
```

---

## 7. 実装の詳細

### 7.1 streamlit_config.py
```python
# MG-ADL項目定義
MGADL_ITEMS = {
    "speech": {"name": "会話", "options": [...], "key": "MGADL_speech"},
    "chewing": {"name": "咀嚼", ...},
    ...
}

# MG Composite項目定義
MGC_ITEMS = {
    "ptosis": {"name": "眼瞼下垂", "values": [0,1,2,3], ...},
    ...
}

# MGQOL項目定義
MGQOL_ITEMS = [
    "MGの病状に不満である",
    ...
]

# ADL→MGC マッピング
ADL_TO_MGC_MAP = {
    "speech": "MGC_speech",
    "chewing": "MGC_chewing",
    ...
}
```

### 7.2 streamlit_predictor.py
```python
class MGPredictor:
    def __init__(self, model_dir='./out/'):
        # モデル・H行列・スケーラー読み込み

    def load_models(self):
        # 5-fold x 4モデル読み込み

    def transform_to_modules(self, patient_data):
        # 33項目 → NMF → 4モジュール

    def predict(self, module_scores):
        # 4モデル x 5-fold 予測
        # アンサンブル計算
        return results
```

### 7.3 streamlit_visualizer.py
```python
def create_radar_chart(patient_scores, mm_avg, non_mm_avg,
                       module_names, ensemble_prob):
    # Plotlyでインタラクティブレーダーチャート作成
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(...))  # non-MM
    fig.add_trace(go.Scatterpolar(...))  # MM
    fig.add_trace(go.Scatterpolar(...))  # 患者
    return fig
```

### 7.4 streamlit_app.py (メイン構造)
```python
import streamlit as st
from streamlit_config import *
from streamlit_predictor import MGPredictor
from streamlit_visualizer import create_radar_chart

# ページ設定
st.set_page_config(page_title="MG予測システム", layout="wide")

# セッション状態初期化
if 'scores' not in st.session_state:
    st.session_state.scores = {...}

# サイドバー: 入力UI
with st.sidebar:
    st.header("📋 患者データ入力")

    # MG-ADL入力
    with st.expander("MG-ADL (8項目)", expanded=True):
        for item_key, item_info in MGADL_ITEMS.items():
            st.session_state.scores[item_key] = st.selectbox(...)
        st.metric("合計", calculate_adl_total())

    # MG-C入力（ADL自動反映機能付き）
    with st.expander("MG Composite (10項目)"):
        ...

    # MGQOL入力
    with st.expander("MGQOL-15r (15項目)"):
        ...

    # 予測ボタン
    predict_btn = st.button("🔮 予測実行", type="primary")
    reset_btn = st.button("🔄 リセット")

# メインエリア
st.title("🏥 MG予測システム")

# 入力サマリー
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("MG-ADL", f"{adl_total}/24")
with col2:
    st.metric("MG Composite", f"{mgc_total}/50")
with col3:
    st.metric("MGQOL-15r", f"{mgqol_total}/30")

# 予測実行
if predict_btn:
    predictor = MGPredictor()
    results = predictor.predict(st.session_state.scores)

    # 結果表示
    st.subheader("🎯 予測結果")

    # モジュールスコア
    st.write("**モジュールスコア**")
    module_df = pd.DataFrame(...)
    st.dataframe(module_df)

    # 各モデル予測
    st.write("**各モデル予測確率**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SVM", f"{results['svm']:.1%}")
    col2.metric("Logistic Reg", f"{results['logreg']:.1%}")
    col3.metric("Random Forest", f"{results['rf']:.1%}")
    col4.metric("Naive Bayes", f"{results['nb']:.1%}")

    # アンサンブル結果
    st.subheader("📊 アンサンブル予測")
    ensemble_prob = results['ensemble']
    status = "MM or better" if ensemble_prob > 0.5 else "non MM"

    st.metric("アンサンブル確率", f"{ensemble_prob:.1%}",
              delta=status, delta_color="normal")

    # レーダーチャート
    st.subheader("📈 モジュールスコア比較")
    fig = create_radar_chart(...)
    st.plotly_chart(fig, use_container_width=True)
```

---

## 8. UI/UX設計

### 8.1 カラースキーム
- **MG-ADL**: 青系 (#007aff)
- **MG Composite**: 緑系 (#34c759)
- **MGQOL-15r**: 紫系 (#af52de)
- **MM予測**: 緑系（良好）、赤系（要注意）

### 8.2 レスポンシブ対応
- サイドバー固定幅: 400px
- メインエリア可変幅
- モバイルではサイドバー折りたたみ

### 8.3 エラーハンドリング
- モデルファイル読み込みエラー → エラーメッセージ表示
- 入力値範囲外 → バリデーション
- 予測失敗 → ログ出力 + ユーザー通知

---

## 9. 実装フェーズ

### Phase 1: 基本機能
- [ ] streamlit_config.py作成（項目定義）
- [ ] streamlit_app.py基本構造
- [ ] 入力UI実装（3スケール）
- [ ] 入力値の集計表示

### Phase 2: 予測機能
- [ ] streamlit_predictor.py作成
- [ ] モデル読み込み機能
- [ ] NMF変換機能
- [ ] 予測実行機能

### Phase 3: 可視化
- [ ] streamlit_visualizer.py作成
- [ ] レーダーチャート実装（Plotly）
- [ ] 結果表示画面

### Phase 4: 最適化
- [ ] セッション管理
- [ ] キャッシング（@st.cache_data）
- [ ] パフォーマンス改善

---

## 10. テスト計画

### 10.1 単体テスト
- 各スケールの点数計算
- NMF変換の正確性
- モデル予測の再現性

### 10.2 統合テスト
- 症例予測.ipynbと同じ入力で同じ結果が出るか
- エンドツーエンドの動作確認

### 10.3 ユースケーステスト
- サンプル患者データ10件で予測実行
- レーダーチャート表示確認

---

## 11. デプロイ

### ローカル実行
```bash
streamlit run streamlit_app.py
```

### Streamlit Cloud（オプション）
- GitHub連携
- secrets.toml設定
- 公開URL発行

---

## 付録A: データマッピング定義

### MG-ADL → 内部形式
```python
{
    "MGADL_speech": adl_scores[0],      # 会話
    "MGADL_chewing": adl_scores[1],     # 咀嚼
    "MGADL_swallowing": adl_scores[2],  # 嚥下
    "MGADL_respiration": adl_scores[3], # 呼吸
    "MGADL_toothbrushing": adl_scores[4], # 歯磨き
    "MGADL_getting_up": adl_scores[5],  # 立ち上がり
    "MGADL_diplopia": adl_scores[6],    # 複視
    "MGADL_ptosis": adl_scores[7]       # 眼瞼下垂
}
```

### MG Composite → 内部形式
```python
{
    "MGC_ptosis": mgc_scores[0],        # 眼瞼下垂
    "MGC_diplopia": mgc_scores[1],      # 複視
    "MGC_eyelid_closure": mgc_scores[2], # 閉眼
    "MGC_speech": mgc_scores[3],        # 会話 ← ADLから
    "MGC_chewing": mgc_scores[4],       # 咀嚼 ← ADLから
    "MGC_swallowing": mgc_scores[5],    # 嚥下 ← ADLから
    "MGC_respiration": mgc_scores[6],   # 呼吸 ← ADLから
    "MGC_neck": mgc_scores[7],          # 頸部
    "MGC_upper_limb": mgc_scores[8],    # 上肢
    "MGC_lower_limb": mgc_scores[9]     # 下肢
}
```

### MGQOL → 内部形式
```python
{
    "MGQOL1_dissatisfaction": mgqol_scores[0],
    "MGQOL2_seeing": mgqol_scores[1],
    ...
    "MGQOL15_dressing": mgqol_scores[14]
}
```

### MG Composite 点数マッピング
```python
MGC_VALUES = {
    "ptosis": [0, 1, 2, 3],           # 4段階
    "diplopia": [0, 1, 3, 4],         # 不規則
    "eyelid_closure": [0, 0, 1, 2],   # 4段階（0が2つ）
    "speech": [0, 2, 4, 6],           # ADLの0-3を変換
    "chewing": [0, 2, 4, 6],          # ADLの0-3を変換
    "swallowing": [0, 2, 5, 6],       # ADLの0-3を変換
    "respiration": [0, 2, 4, 9],      # ADLの0-3を変換
    "neck": [0, 1, 3, 4],
    "upper_limb": [0, 2, 4, 5],
    "lower_limb": [0, 2, 4, 5]
}
```

---

## 付録B: スクリーンショット（イメージ）

```
┌──────────────────────────────────────────────────────────────┐
│ 🏥 MG予測システム                                            │
├──────────────┬───────────────────────────────────────────────┤
│ 📋 患者データ│                                               │
│              │  ╔═══════════════════════════════════════════╗│
│ ▼ MG-ADL     │  ║  📊 入力サマリー                          ║│
│  会話: 0     │  ║  ┌────────┬────────┬─────────┐           ║│
│  咀嚼: 0     │  ║  │ MG-ADL │  MG-C  │ MGQOL   │           ║│
│  嚥下: 0     │  ║  │  0/24  │  0/50  │  0/30   │           ║│
│  ...         │  ║  └────────┴────────┴─────────┘           ║│
│  合計: 0/24  │  ╚═══════════════════════════════════════════╝│
│              │                                               │
│ ▶ MG-C       │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│  ...         │  （予測実行前は表示なし）                    │
│              │                                               │
│ ▶ MGQOL      │                                               │
│  ...         │                                               │
│              │                                               │
│ ┌──────────┐│                                               │
│ │🔮 予測実行││                                               │
│ └──────────┘│                                               │
│ [🔄リセット] │                                               │
└──────────────┴───────────────────────────────────────────────┘

予測後:

┌──────────────────────────────────────────────────────────────┐
│ 🎯 予測結果                                                  │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ モジュールスコア                                        │ │
│ │  QOL: 0.0391  Diplopia: 0.0000  Ptosis: 0.0000         │ │
│ │  Systemic: 0.0280                                      │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                              │
│ 各モデル予測                                                │
│ ┌──────┬──────┬──────┬──────┐                              │
│ │ SVM  │LogReg│  RF  │  NB  │                              │
│ │93.2% │61.7% │94.7% │98.8% │                              │
│ └──────┴──────┴──────┴──────┘                              │
│                                                              │
│ ╔═══════════════════════════════════╗                       │
│ ║ アンサンブル予測: 87.1%          ║                       │
│ ║ 分類結果: MM or better            ║                       │
│ ╚═══════════════════════════════════╝                       │
│                                                              │
│ 📈 モジュールスコア比較                                     │
│     QOL                                                      │
│      │\                                                     │
│      │  \  患者                                             │
│      │    ●────MM群                                         │
│      │      \  /                                            │
│ Systemic───┼───Diplopia                                    │
│      │      /  \                                            │
│      │    ●────non-MM群                                     │
│      │  /                                                   │
│    Ptosis                                                    │
└──────────────────────────────────────────────────────────────┘
```

---

以上
