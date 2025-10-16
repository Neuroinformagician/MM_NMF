"""
MG Scale iOS-style Component for Streamlit
完全にmg-scale-iOS.htmlのUIを再現
"""

import streamlit as st
import streamlit.components.v1 as components
import json

def create_mg_scale_component(scale_type="adl", items=None):
    """
    mg-scale-iOS.htmlと同じUIを生成するコンポーネント
    """

    if scale_type == "adl":
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
                    margin: 0;
                    padding: 10px;
                    background-color: #f2f2f7;
                    color: #1c1c1e;
                }

                .tap-input-section {
                    background-color: #f9f9fb;
                    border-radius: 10px;
                    padding: 15px;
                    margin-bottom: 20px;
                    border: 1px solid #e5e5ea;
                }

                .tap-input-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 10px;
                }

                .tap-item {
                    background-color: white;
                    border: 2px solid #e5e5ea;
                    border-radius: 10px;
                    padding: 12px;
                    cursor: pointer;
                    transition: all 0.2s ease;
                    user-select: none;
                    -webkit-tap-highlight-color: transparent;
                    position: relative;
                }

                .tap-item:active {
                    transform: scale(0.98);
                }

                .tap-item-name {
                    font-weight: 600;
                    font-size: 0.9rem;
                    margin-bottom: 5px;
                }

                .tap-item-score {
                    font-size: 1.5rem;
                    font-weight: 700;
                    text-align: center;
                    padding: 5px;
                    border-radius: 5px;
                    background-color: #f2f2f7;
                    color: #007aff;
                }

                .tap-item.has-score {
                    border-color: #007aff;
                    background-color: rgba(0, 122, 255, 0.05);
                }

                .total-display {
                    background: #007aff;
                    color: white;
                    font-size: 1.3rem;
                    font-weight: 700;
                    text-align: center;
                    padding: 16px;
                    border-radius: 10px;
                    margin: 20px 0;
                }

                .tap-instruction {
                    background-color: #fff3cd;
                    border: 1px solid #ffc107;
                    color: #856404;
                    padding: 10px;
                    border-radius: 8px;
                    margin-bottom: 15px;
                    font-size: 0.85rem;
                    text-align: center;
                }
            </style>
        </head>
        <body>
            <div class="tap-instruction">
                タップで点数入力：1回タップ→1点、2回→2点、3回→3点、4回→0点に戻る
            </div>

            <div class="tap-input-section">
                <div class="tap-input-grid" id="adl-tap-grid"></div>
            </div>

            <div id="adl-total" class="total-display">合計点: 0/24点</div>

            <script>
                const adlItems = %s;
                let scores = new Array(adlItems.length).fill(0);

                function updateTotal() {
                    const total = scores.reduce((sum, score) => sum + score, 0);
                    document.getElementById('adl-total').textContent = `合計点: ${total}/24点`;

                    // Streamlitに値を送信
                    const data = {
                        scores: scores,
                        total: total
                    };
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: data
                    }, '*');
                }

                function createTapItem(item, index) {
                    const div = document.createElement('div');
                    div.className = 'tap-item';
                    div.dataset.index = index;

                    const nameDiv = document.createElement('div');
                    nameDiv.className = 'tap-item-name';
                    nameDiv.textContent = item.name;

                    const scoreDiv = document.createElement('div');
                    scoreDiv.className = 'tap-item-score';
                    scoreDiv.textContent = '0';

                    div.appendChild(nameDiv);
                    div.appendChild(scoreDiv);

                    div.addEventListener('click', function() {
                        scores[index] = (scores[index] + 1) %% 4;
                        scoreDiv.textContent = scores[index];

                        if (scores[index] > 0) {
                            div.classList.add('has-score');
                        } else {
                            div.classList.remove('has-score');
                        }

                        updateTotal();
                    });

                    return div;
                }

                // アイテムを生成
                const grid = document.getElementById('adl-tap-grid');
                adlItems.forEach((item, index) => {
                    grid.appendChild(createTapItem(item, index));
                });

                // 初期値を設定
                updateTotal();
            </script>
        </body>
        </html>
        """ % json.dumps(items)

    elif scale_type == "mgc":
        # MGC用のHTML（同様に実装）
        html_content = create_mgc_html(items)
    elif scale_type == "mgqol":
        # MGQOL用のHTML（同様に実装）
        html_content = create_mgqol_html(items)
    else:
        html_content = "<div>Invalid scale type</div>"

    return html_content

def create_mgc_html(items):
    """MGC用のHTML生成"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
                margin: 0;
                padding: 10px;
                background-color: #f2f2f7;
                color: #1c1c1e;
            }}

            .tap-input-section {{
                background-color: #f9f9fb;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 20px;
                border: 1px solid #e5e5ea;
            }}

            .tap-input-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 10px;
            }}

            .tap-item {{
                background-color: white;
                border: 2px solid #e5e5ea;
                border-radius: 10px;
                padding: 12px;
                cursor: pointer;
                transition: all 0.2s ease;
                user-select: none;
                -webkit-tap-highlight-color: transparent;
                position: relative;
            }}

            .tap-item:active {{
                transform: scale(0.98);
            }}

            .tap-item-name {{
                font-weight: 600;
                font-size: 0.8rem;
                margin-bottom: 5px;
            }}

            .tap-item-score {{
                font-size: 1.5rem;
                font-weight: 700;
                text-align: center;
                padding: 5px;
                border-radius: 5px;
                background-color: #f2f2f7;
                color: #34c759;
            }}

            .tap-item.has-score {{
                border-color: #34c759;
                background-color: rgba(52, 199, 89, 0.05);
            }}

            .total-display {{
                background: #34c759;
                color: white;
                font-size: 1.3rem;
                font-weight: 700;
                text-align: center;
                padding: 16px;
                border-radius: 10px;
                margin: 20px 0;
            }}

            .tap-instruction {{
                background-color: #fff3cd;
                border: 1px solid #ffc107;
                color: #856404;
                padding: 10px;
                border-radius: 8px;
                margin-bottom: 15px;
                font-size: 0.85rem;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="tap-instruction">
            タップで点数入力：タップ回数で各項目の点数が変わります
        </div>

        <div class="tap-input-section">
            <div class="tap-input-grid" id="mgc-tap-grid"></div>
        </div>

        <div id="mgc-total" class="total-display">合計点: 0/50点</div>

        <script>
            const mgcItems = {json.dumps(items)};
            let scores = new Array(mgcItems.length).fill(0);

            function updateTotal() {{
                let total = 0;
                mgcItems.forEach((item, index) => {{
                    total += item.values[scores[index]];
                }});
                document.getElementById('mgc-total').textContent = `合計点: ${{total}}/50点`;

                // Streamlitに値を送信
                const data = {{
                    scores: scores,
                    total: total
                }};
                window.parent.postMessage({{
                    type: 'streamlit:setComponentValue',
                    value: data
                }}, '*');
            }}

            function createTapItem(item, index) {{
                const div = document.createElement('div');
                div.className = 'tap-item';
                div.dataset.index = index;

                const nameDiv = document.createElement('div');
                nameDiv.className = 'tap-item-name';
                nameDiv.textContent = item.name;

                const scoreDiv = document.createElement('div');
                scoreDiv.className = 'tap-item-score';
                scoreDiv.textContent = item.values[0];

                div.appendChild(nameDiv);
                div.appendChild(scoreDiv);

                div.addEventListener('click', function() {{
                    scores[index] = (scores[index] + 1) % item.values.length;
                    scoreDiv.textContent = item.values[scores[index]];

                    if (scores[index] > 0) {{
                        div.classList.add('has-score');
                    }} else {{
                        div.classList.remove('has-score');
                    }}

                    updateTotal();
                }});

                return div;
            }}

            // アイテムを生成
            const grid = document.getElementById('mgc-tap-grid');
            mgcItems.forEach((item, index) => {{
                grid.appendChild(createTapItem(item, index));
            }});

            // 初期値を設定
            updateTotal();
        </script>
    </body>
    </html>
    """

def create_mgqol_html(items):
    """MGQOL用のHTML生成"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
                margin: 0;
                padding: 10px;
                background-color: #f2f2f7;
                color: #1c1c1e;
            }}

            .tap-input-section {{
                background-color: #f9f9fb;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 20px;
                border: 1px solid #e5e5ea;
            }}

            .tap-input-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 10px;
            }}

            .tap-item {{
                background-color: white;
                border: 2px solid #e5e5ea;
                border-radius: 10px;
                padding: 12px;
                cursor: pointer;
                transition: all 0.2s ease;
                user-select: none;
                -webkit-tap-highlight-color: transparent;
                position: relative;
            }}

            .tap-item:active {{
                transform: scale(0.98);
            }}

            .tap-item-name {{
                font-weight: 600;
                font-size: 0.8rem;
                margin-bottom: 5px;
            }}

            .tap-item-score {{
                font-size: 1.5rem;
                font-weight: 700;
                text-align: center;
                padding: 5px;
                border-radius: 5px;
                background-color: #f2f2f7;
                color: #af52de;
            }}

            .tap-item.has-score {{
                border-color: #af52de;
                background-color: rgba(175, 82, 222, 0.05);
            }}

            .total-display {{
                background: #af52de;
                color: white;
                font-size: 1.3rem;
                font-weight: 700;
                text-align: center;
                padding: 16px;
                border-radius: 10px;
                margin: 20px 0;
            }}

            .tap-instruction {{
                background-color: #fff3cd;
                border: 1px solid #ffc107;
                color: #856404;
                padding: 10px;
                border-radius: 8px;
                margin-bottom: 15px;
                font-size: 0.85rem;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="tap-instruction">
            タップで点数入力：1回タップ→1点、2回→2点、3回→0点に戻る
        </div>

        <div class="tap-input-section">
            <div class="tap-input-grid" id="mgqol-tap-grid"></div>
        </div>

        <div id="mgqol-total" class="total-display">合計点: 0/30点</div>

        <script>
            const mgqolItems = {json.dumps(items)};
            let scores = new Array(mgqolItems.length).fill(0);

            function updateTotal() {{
                const total = scores.reduce((sum, score) => sum + score, 0);
                document.getElementById('mgqol-total').textContent = `合計点: ${{total}}/30点`;

                // Streamlitに値を送信
                const data = {{
                    scores: scores,
                    total: total
                }};
                window.parent.postMessage({{
                    type: 'streamlit:setComponentValue',
                    value: data
                }}, '*');
            }}

            function createTapItem(item, index) {{
                const div = document.createElement('div');
                div.className = 'tap-item';
                div.dataset.index = index;

                const nameDiv = document.createElement('div');
                nameDiv.className = 'tap-item-name';
                nameDiv.textContent = `${{index + 1}}. ${{item.name.substring(0, 10)}}...`;
                nameDiv.title = item.name;

                const scoreDiv = document.createElement('div');
                scoreDiv.className = 'tap-item-score';
                scoreDiv.textContent = '0';

                div.appendChild(nameDiv);
                div.appendChild(scoreDiv);

                div.addEventListener('click', function() {{
                    scores[index] = (scores[index] + 1) % 3;
                    scoreDiv.textContent = scores[index];

                    if (scores[index] > 0) {{
                        div.classList.add('has-score');
                    }} else {{
                        div.classList.remove('has-score');
                    }}

                    updateTotal();
                }});

                return div;
            }}

            // アイテムを生成
            const grid = document.getElementById('mgqol-tap-grid');
            mgqolItems.forEach((item, index) => {{
                grid.appendChild(createTapItem(item, index));
            }});

            // 初期値を設定
            updateTotal();
        </script>
    </body>
    </html>
    """