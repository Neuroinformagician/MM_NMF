@echo off
REM ローカルで MG MM 判定アプリ（医療調 app_pro.py）を開く。
REM 使い方: このファイルをダブルクリック、または  run_local.bat
cd /d "%~dp0"

if not exist venv (
  echo ==^> 仮想環境を作成しています...
  python -m venv venv
)
call venv\Scripts\activate

echo ==^> 依存パッケージを確認/インストールしています...
pip install -q -r requirements.txt

echo ==^> アプリを起動します。ブラウザが自動で開きます。停止は Ctrl+C。
streamlit run app_pro.py
