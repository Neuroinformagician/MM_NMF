#!/usr/bin/env bash
# ローカルで MG MM 判定アプリ（医療調 app_pro.py）を開く。
# 使い方: ターミナルで  ./run_local.sh
set -e
cd "$(dirname "$0")"

# 仮想環境（初回のみ作成）
if [ ! -d "venv" ]; then
  echo "==> 仮想環境を作成しています..."
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

# 依存インストール（毎回。既に入っていれば一瞬）
echo "==> 依存パッケージを確認/インストールしています..."
pip install -q -r requirements.txt

# 起動（自動でブラウザが開きます。開かなければ http://localhost:8501）
echo "==> アプリを起動します。ブラウザが自動で開きます。停止は Ctrl+C。"
streamlit run app_pro.py
