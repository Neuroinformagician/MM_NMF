# MG MM判定システム

重症筋無力症（MG）患者の治療反応性（MM or better）を予測するWebアプリケーション

## 概要

このシステムは、以下の3つのスコアから患者の治療反応性を予測します：
- **MGC** (Myasthenia Gravis Composite): 身体機能評価
- **MGADL** (MG Activities of Daily Living): 日常生活動作評価
- **MGQOL15** (MG Quality of Life): 生活の質評価

## 技術的特徴

- **NMF（非負値行列因子分解）**: 33項目を4つのモジュールに次元削減
  - module QOL: 生活の質関連
  - module Diplopia: 複視関連
  - module Ptosis: 眼瞼下垂関連
  - module Systemic: 全身症状関連

- **機械学習モデル**: 5-fold交差検証による4つのアルゴリズム
  - SVM（サポートベクターマシン）
  - ロジスティック回帰
  - ランダムフォレスト
  - ナイーブベイズ

## インストール

```bash
# 必要なパッケージをインストール
pip install -r requirements.txt
```

## データ準備

1. まず学習済みモデルを作成：
```bash
python step1_load_data_and_NMF.py
python step2_prediction_with_NMF.py
```

2. 以下のファイルが`out/`フォルダに生成されることを確認：
- `nmf_model.pkl`
- `H_matrix.pkl`
- `svm_fold_0.pkl` ~ `svm_fold_4.pkl`
- `logistic_regression_fold_0.pkl` ~ `logistic_regression_fold_4.pkl`
- `random_forest_fold_0.pkl` ~ `random_forest_fold_4.pkl`
- `naive_bayes_fold_0.pkl` ~ `naive_bayes_fold_4.pkl`

## 使用方法

1. Streamlitアプリを起動：
```bash
streamlit run app_simple.py
```

> `app_simple.py` は、学習時に保存した `out/minmax_scaler.pkl` を使って正しく正規化し、
> NMF → 4モデル×5foldの平均確率 → 各モデルのROC最適カットオフで判定する、
> シンプルで自己完結した1ファイル版です。

2. ブラウザで自動的に開くWebページで：
   - 各項目のスコアをスライダーで入力
   - 「予測を実行」ボタンをクリック
   - 結果を確認

## 結果の解釈

- **MM or better**: 治療反応性良好（確率 ≥ 50%）
- **non-MM**: 治療反応性不良（確率 < 50%）

**MM (Minimal Manifestations)**: わずかな症状のみが残存している状態

## ファイル構成

```
MM_NMF/
├── data/
│   ├── df_3rd.csv
│   └── df_4th.csv
├── out/
│   └── [学習済みモデル]
├── fig/
│   └── [グラフ出力]
├── step1_load_data_and_NMF.py
├── step2_prediction_with_NMF.py
├── app.py                 # Streamlitアプリ
├── requirements.txt
└── README.md
```

## 注意事項

- このシステムは研究目的で開発されたものです
- 臨床での使用には医師の判断が必要です
- 予測結果は参考情報として扱ってください