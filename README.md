# MG予測システム（MM_NMF）

重症筋無力症（MG）患者のMM（Minimal Manifestations）予測システム

## 🚀 アプリを試す

**[https://mmnmfmg.streamlit.app/](https://mmnmfmg.streamlit.app/)**

## 概要

このシステムは、3つの臨床評価スケール（MG-ADL、MG Composite、MGQOL-15r）から、NMFと機械学習を用いてMG患者のMM状態を予測します。

### 主な機能

- **ボタンタップ入力**: 各項目を簡単に入力（0→1→2→3と循環）
- **自動反映**: MG-ADLの一部項目がMG Compositeに自動反映
- **NMF変換**: 33次元の臨床スコアを4次元のモジュールスコアに次元削減
- **3モデルアンサンブル**: SVM、Random Forest、Naive Bayesによるソフト投票
- **視覚化**: Plotlyレーダーチャートで患者とMM群/non-MM群を比較

## 技術スタック

- **Streamlit**: Webアプリフレームワーク
- **scikit-learn**: 機械学習（NMF、SVM、Random Forest、Naive Bayes）
- **Plotly**: インタラクティブチャート
- **NumPy/Pandas**: データ処理

## ドキュメント

詳細は[README_STREAMLIT.md](README_STREAMLIT.md)をご覧ください。

## ローカルで実行

```bash
pip install -r requirements.txt
streamlit run streamlit_app_simple.py
```

## ライセンス

(プロジェクトのライセンスに準ずる)
