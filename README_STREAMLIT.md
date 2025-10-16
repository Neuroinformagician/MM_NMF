# MG予測システム - Streamlitアプリ

重症筋無力症（MG）患者のMM（Minimal Manifestations）予測システム

## 機能

- **3つの評価スケール入力**
  - MG-ADL (8項目、0-24点)
  - MG Composite (10項目、0-50点)
  - MGQOL-15r (15項目、0-30点)
  - ボタンタップで簡単入力
  - MG-ADLからMG Compositeへの自動反映

- **機械学習による予測**
  - NMFによる4モジュールスコア算出
  - 3モデル（SVM、Random Forest、Naive Bayes）× 5-fold
  - 最適カットオフによるSoft Voting
  - MM/non-MM分類

- **インタラクティブな可視化**
  - 各モデルの予測確率と判定
  - Plotlyレーダーチャート
  - 患者 vs MM群 vs non-MM群の比較

## ローカルで実行

### 1. 依存ライブラリのインストール

```bash
pip install -r requirements_streamlit.txt
```

または

```bash
pip install streamlit numpy pandas scikit-learn plotly matplotlib
```

### 2. アプリの起動

```bash
streamlit run streamlit_app_simple.py
```

ブラウザが自動で開き、`http://localhost:8501` でアプリが表示されます。

### 3. 使い方

1. **MG-ADL入力**: ボタンをクリックしてスコアを入力（0→1→2→3と循環）
   - ⚡マークの項目（会話・咀嚼・嚥下・呼吸）はMG Compositeに自動反映
   - 「MGCへ →」で次のステップへ

2. **MG Composite入力**: 10項目を入力
   - ⚡マーク付き項目はMG-ADLから自動反映済み
   - 「MGQOLへ →」で次のステップへ

3. **MGQOL-15r入力**: 15項目を入力（各0-2点）
   - 「予測実行」ボタンをクリック

4. **結果確認**
   - 予測結果（MM or better / non MM）
   - 3モデルの投票結果
   - 入力スコア合計
   - 各モデルの予測確率
   - モジュールスコア
   - レーダーチャート比較

## Streamlit Cloudへのデプロイ

### 前提条件
- GitHubアカウント
- Streamlit Cloudアカウント（無料、GitHubでログイン可能）

### デプロイ手順

#### 1. GitHubにプッシュ

```bash
# まだコミットしていない場合
git add .
git commit -m "Add Streamlit MG prediction app"
git push origin main_rev
```

#### 2. Streamlit Cloudでデプロイ

1. https://share.streamlit.io にアクセス
2. 「New app」をクリック
3. 設定を入力：
   - **Repository**: あなたのGitHubリポジトリを選択
   - **Branch**: `main_rev`
   - **Main file path**: `streamlit_app_simple.py`
   - **Python version**: 3.9以上を選択
4. 「Deploy!」をクリック

#### 3. デプロイ完了

数分でデプロイが完了し、公開URLが発行されます。
例: `https://your-app-name.streamlit.app`

### 注意事項

- **データファイル**: `data/` と `out/` ディレクトリがリポジトリに含まれていることを確認
- **モデルサイズ**: 合計約13MBのモデルファイルが含まれます
- **無料プラン**: Streamlit Cloudの無料プランで十分動作します

## ファイル構成

```
.
├── streamlit_app_simple.py   # メインアプリケーション（本番用）
├── streamlit_config.py       # 項目定義・設定
├── streamlit_predictor.py    # 予測ロジック（NMF + ML）
├── streamlit_visualizer.py   # 可視化ロジック（Plotly）
├── requirements_streamlit.txt # 依存ライブラリ
├── README_STREAMLIT.md       # このファイル
├── data/
│   └── df_4th.csv           # 元データ（スケーラー学習用）
└── out/
    ├── H_matrix.pkl          # NMF H行列
    ├── W_MM.pkl              # MM群/non-MM群データ
    ├── module_names_reordered.pkl  # モジュール名
    ├── optimal_cutoffs.pkl   # 最適カットオフ
    └── *_fold_*.pkl          # 学習済みモデル（15ファイル: 3モデル × 5-fold）
```

## トラブルシューティング

### モデルファイルが見つからない

```
FileNotFoundError: 以下の必須ファイルが見つかりません
```

→ `data/` と `out/` ディレクトリがgitにコミットされているか確認

### バージョン不整合

```
ModuleNotFoundError: No module named 'streamlit'
```

→ `pip install -r requirements_streamlit.txt` を実行

### Plotlyチャートが表示されない

→ ブラウザのキャッシュをクリアして再読み込み

## 技術スタック

- **Streamlit 1.28+**: Webアプリフレームワーク
- **Plotly 5.17+**: インタラクティブチャート
- **scikit-learn 1.3+**: 機械学習
- **NumPy/Pandas**: データ処理

## ライセンス

(プロジェクトのライセンスに準ずる)

## 技術詳細

### 予測アルゴリズム

1. **NMF（Non-negative Matrix Factorization）**
   - 33次元の臨床スコアを4次元のモジュールスコアに次元削減
   - モジュール: QOL、Diplopia、Ptosis、Systemic

2. **機械学習モデル**
   - SVM、Random Forest、Naive Bayes の3モデル
   - 各モデルで5-fold交差検証
   - 最適カットオフによるSoft Voting

3. **最終判定**
   - 各モデルが独立に判定（最適カットオフ使用）
   - 3モデル中2つ以上が「MM or better」なら最終判定もMM or better

## 変更履歴

- v1.0 (2025-10-16): 初版リリース
  - ボタンタップ入力UI
  - Soft Voting判定
  - 3モデルアンサンブル
  - レーダーチャート可視化
  - ステップバイステップ入力
