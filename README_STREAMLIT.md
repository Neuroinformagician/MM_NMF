# MG予測システム - Streamlitアプリ

重症筋無力症（MG）患者のMM（Minimal Manifestations）予測システム

## 機能

- **3つの評価スケール入力**
  - MG-ADL (8項目、0-24点)
  - MG Composite (10項目、0-50点)
  - MGQOL-15r (15項目、0-30点)

- **機械学習による予測**
  - NMFによる4モジュールスコア算出
  - 4モデル × 5-fold のアンサンブル予測
  - MM/non-MM分類

- **インタラクティブな可視化**
  - Plotlyレーダーチャート
  - 患者 vs MM群 vs non-MM群の比較
  - モジュール別詳細評価

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
streamlit run streamlit_app.py
```

ブラウザが自動で開き、`http://localhost:8501` でアプリが表示されます。

### 3. 使い方

1. 左サイドバーから患者データを入力
   - MG-ADL: 日常生活動作評価
   - MG Composite: 総合評価（ADLから自動反映可能）
   - MGQOL-15r: QOL評価

2. 「🔮 予測実行」ボタンをクリック

3. 結果確認
   - モジュールスコア
   - 各モデル予測確率
   - アンサンブル予測結果
   - レーダーチャート比較

## Streamlit Cloudへのデプロイ

### 方法1: GitHub経由

1. このリポジトリをGitHubにpush
2. https://share.streamlit.io にアクセス
3. リポジトリを選択
4. Main file: `streamlit_app.py`
5. Python version: 3.9以上
6. Deploy!

### 方法2: 直接デプロイ

Streamlit Cloudの管理画面から：
- Repository: このリポジトリのURL
- Branch: `main_rev`
- Main file path: `streamlit_app.py`

## ファイル構成

```
.
├── streamlit_app.py          # メインアプリケーション
├── streamlit_config.py       # 項目定義・設定
├── streamlit_predictor.py    # 予測ロジック
├── streamlit_visualizer.py   # 可視化ロジック
├── requirements_streamlit.txt # 依存ライブラリ
├── data/
│   └── df_4th.csv           # 元データ（スケーラー学習用）
└── out/
    ├── H_matrix.pkl          # NMF行列
    ├── W_MM.pkl              # MM群データ
    ├── module_names_reordered.pkl
    └── *_fold_*.pkl          # 学習済みモデル（20ファイル）
```

## サンプルデータ

アプリ起動後、「📝 サンプルデータを読み込む」ボタンで、
`症例予測.ipynb`と同じサンプル患者データを読み込めます。

**期待される結果:**
- モジュールスコア: QOL=0.0391, Systemic=0.0280
- アンサンブル予測: 87.1% (MM or better)

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

## 開発

### 変更履歴

- v1.0 (2025-10-16): 初版リリース
  - 基本機能実装
  - 3スケール入力UI
  - アンサンブル予測
  - レーダーチャート可視化
