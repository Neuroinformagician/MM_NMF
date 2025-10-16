"""
Streamlit MG Prediction App - Configuration
設定ファイル: 項目定義、パス設定、データマッピング
"""

import os
from pathlib import Path

# ============================================================================
# パス設定
# ============================================================================

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "out"
FIG_DIR = BASE_DIR / "fig"

# 必須ファイル
REQUIRED_FILES = {
    "original_data": DATA_DIR / "df_4th.csv",
    "H_matrix": MODEL_DIR / "H_matrix.pkl",
    "W_MM": MODEL_DIR / "W_MM.pkl",
    "module_names": MODEL_DIR / "module_names_reordered.pkl"
}

# モデルファイル
MODEL_FILES = {
    "SVM": [MODEL_DIR / f"svm_fold_{i}.pkl" for i in range(5)],
    "Logistic Regression": [MODEL_DIR / f"logistic_regression_fold_{i}.pkl" for i in range(5)],
    "Random Forest": [MODEL_DIR / f"random_forest_fold_{i}.pkl" for i in range(5)],
    "Naive Bayes": [MODEL_DIR / f"naive_bayes_fold_{i}.pkl" for i in range(5)]
}

# ============================================================================
# MG-ADL 項目定義
# ============================================================================

MGADL_ITEMS = [
    {
        "key": "speech",
        "name": "会話",
        "options": [
            "0: 正常",
            "1: 間欠的に不明瞭もしくは鼻声",
            "2: 常に不明瞭もしくは鼻声、しかし聞いて理解可能",
            "3: 聞いて理解するのが困難"
        ],
        "internal_key": "MGADL_speech"
    },
    {
        "key": "chewing",
        "name": "咀嚼",
        "options": [
            "0: 正常",
            "1: 固形物で疲労",
            "2: 柔らかい食物で疲労",
            "3: 経管栄養"
        ],
        "internal_key": "MGADL_chewing"
    },
    {
        "key": "swallowing",
        "name": "嚥下",
        "options": [
            "0: 正常",
            "1: まれにむせる",
            "2: 頻回にむせるため、食事の変更が必要",
            "3: 経管栄養"
        ],
        "internal_key": "MGADL_swallowing"
    },
    {
        "key": "respiration",
        "name": "呼吸",
        "options": [
            "0: 正常",
            "1: 体動時の息切れ",
            "2: 安静時の息切れ",
            "3: 人工呼吸を要する"
        ],
        "internal_key": "MGADL_respiration"
    },
    {
        "key": "toothbrushing",
        "name": "歯磨き・櫛使用の障害",
        "options": [
            "0: なし",
            "1: 努力を要するが休息を要しない",
            "2: 休息を要する",
            "3: できない"
        ],
        "internal_key": "MGADL_toothbrushing"
    },
    {
        "key": "getting_up",
        "name": "椅子からの立ち上がり障害",
        "options": [
            "0: なし",
            "1: 軽度、時々腕を使う",
            "2: 中等度、常に腕を使う",
            "3: 高度、介助を要する"
        ],
        "internal_key": "MGADL_getting_up"
    },
    {
        "key": "diplopia",
        "name": "複視",
        "options": [
            "0: なし",
            "1: あるが毎日ではない",
            "2: 毎日起こるが持続的でない",
            "3: 常にある"
        ],
        "internal_key": "MGADL_diplopia"
    },
    {
        "key": "ptosis",
        "name": "眼瞼下垂",
        "options": [
            "0: なし",
            "1: あるが毎日ではない",
            "2: 毎日起こるが持続的でない",
            "3: 常にある"
        ],
        "internal_key": "MGADL_ptosis"
    }
]

# ============================================================================
# MG Composite 項目定義
# ============================================================================

MGC_ITEMS = [
    {
        "key": "ptosis",
        "name": "上方視時の眼瞼下垂出現までの時間",
        "description": "（医師の診察）",
        "options": [
            "0点: >45秒",
            "1点: 11〜45秒",
            "2点: 1〜10秒",
            "3点: 常時"
        ],
        "values": [0, 1, 2, 3],
        "internal_key": "MGC_ptosis"
    },
    {
        "key": "diplopia",
        "name": "側方視時の複視出現までの時間",
        "description": "（医師の診察）",
        "options": [
            "0点: >45秒",
            "1点: 11〜45秒",
            "3点: 1〜10秒",
            "4点: 常時"
        ],
        "values": [0, 1, 3, 4],
        "internal_key": "MGC_diplopia"
    },
    {
        "key": "eyelid_closure",
        "name": "閉眼の筋力",
        "description": "（医師の診察）",
        "options": [
            "0点: 正常",
            "0点: 軽度低下（閉眼維持可能）",
            "1点: 中等度低下（閉眼維持困難）",
            "2点: 重度低下（閉眼不能）"
        ],
        "values": [0, 0, 1, 2],
        "internal_key": "MGC_eyelid_closure"
    },
    {
        "key": "speech",
        "name": "会話、発音",
        "description": "（MGADL）",
        "options": [
            "0点: 正常",
            "2点: 時に不明瞭または鼻声",
            "4点: 常に不明瞭または鼻声だが理解可能",
            "6点: 不明瞭で理解が困難"
        ],
        "values": [0, 2, 4, 6],
        "internal_key": "MGC_speech",
        "from_adl": "speech"
    },
    {
        "key": "chewing",
        "name": "咬む動作",
        "description": "（MGADL）",
        "options": [
            "0点: 正常",
            "2点: 固い食物で疲労",
            "4点: 柔らかい食物でも疲労",
            "6点: 栄養チューブ使用"
        ],
        "values": [0, 2, 4, 6],
        "internal_key": "MGC_chewing",
        "from_adl": "chewing"
    },
    {
        "key": "swallowing",
        "name": "飲み込み動作",
        "description": "（MGADL）",
        "options": [
            "0点: 正常",
            "2点: まれにむせる",
            "5点: 頻回のむせのため食事に工夫を要す",
            "6点: 栄養チューブ使用"
        ],
        "values": [0, 2, 5, 6],
        "internal_key": "MGC_swallowing",
        "from_adl": "swallowing"
    },
    {
        "key": "respiration",
        "name": "MGによる呼吸状態",
        "description": "（MGADL）",
        "options": [
            "0点: 正常",
            "2点: 活動時息切れ",
            "4点: 安静時息切れ",
            "9点: 呼吸補助装置使用"
        ],
        "values": [0, 2, 4, 9],
        "internal_key": "MGC_respiration",
        "from_adl": "respiration"
    },
    {
        "key": "neck",
        "name": "頸の前屈/背屈筋力",
        "description": "（弱い方を選択、医師の診察）",
        "options": [
            "0点: 正常",
            "1点: 軽度低下",
            "3点: 中等度低下（おおよそ半減）",
            "4点: 重度低下"
        ],
        "values": [0, 1, 3, 4],
        "internal_key": "MGC_neck"
    },
    {
        "key": "upper_limb",
        "name": "上肢の挙上筋力",
        "description": "（医師の診察）",
        "options": [
            "0点: 正常",
            "2点: 軽度低下",
            "4点: 中等度低下（おおよそ半減）",
            "5点: 重度低下"
        ],
        "values": [0, 2, 4, 5],
        "internal_key": "MGC_upper_limb"
    },
    {
        "key": "lower_limb",
        "name": "下肢の挙上筋力",
        "description": "（医師の診察）",
        "options": [
            "0点: 正常",
            "2点: 軽度低下",
            "4点: 中等度低下（おおよそ半減）",
            "5点: 重度低下"
        ],
        "values": [0, 2, 4, 5],
        "internal_key": "MGC_lower_limb"
    }
]

# ============================================================================
# MGQOL-15r 項目定義
# ============================================================================

MGQOL_ITEMS = [
    {"key": "q1", "name": "MGの病状に不満である", "internal_key": "MGQOL1_dissatisfaction"},
    {"key": "q2", "name": "MGのため物を見る際に支障が生じる（二重に見える、まぶたが下がる、など）", "internal_key": "MGQOL2_seeing"},
    {"key": "q3", "name": "MGのため食べる際に支障が生じる", "internal_key": "MGQOL3_eating"},
    {"key": "q4", "name": "MGのため社会活動に制限が生じている", "internal_key": "MGQOL4_social_activity_restriction"},
    {"key": "q5", "name": "MGのため趣味や娯楽を以前ほど楽しめない", "internal_key": "MGQOL5_hobby_entertainment"},
    {"key": "q6", "name": "MGのため家族の要求に十分応えられない", "internal_key": "MGQOL6_family_role"},
    {"key": "q7", "name": "MGのため行動に工夫が必要", "internal_key": "MGQOL7_behavior_modification"},
    {"key": "q8", "name": "MGのため仕事や役割（家庭を含む）に制限が生じ悩まされている", "internal_key": "MGQOL8_work_impact"},
    {"key": "q9", "name": "MGのため話す際に支障が生じる", "internal_key": "MGQOL9_speaking"},
    {"key": "q10", "name": "MGのため外出、用足しをひとりですることが難しい（車の運転、買い物など）", "internal_key": "MGQOL10_driving"},
    {"key": "q11", "name": "MGのため気持ちが落ち込む", "internal_key": "MGQOL11_feeling_down"},
    {"key": "q12", "name": "MGのため歩行に支障が生じる", "internal_key": "MGQOL12_walking"},
    {"key": "q13", "name": "MGのため周囲と同じ早さで行動出来ない（公共の場所などで）", "internal_key": "MGQOL13_quick_action"},
    {"key": "q14", "name": "MGでつらくて精神的に押し潰されそうになる", "internal_key": "MGQOL14_mental_crushing"},
    {"key": "q15", "name": "MGのため身支度に支障が生じる", "internal_key": "MGQOL15_dressing"}
]

MGQOL_OPTIONS = [
    "0: 全くそうは思わない",
    "1: 少しそう思う",
    "2: 強くそう思う"
]

# ============================================================================
# 内部データ形式のカラム順序（df_4th.csvと同じ）
# ============================================================================

DF_COLUMNS = [
    "MGC_ptosis", "MGC_diplopia", "MGC_eyelid_closure", "MGC_speech", "MGC_chewing",
    "MGC_swallowing", "MGC_respiration", "MGC_neck", "MGC_upper_limb", "MGC_lower_limb",
    "MGADL_speech", "MGADL_chewing", "MGADL_swallowing", "MGADL_respiration",
    "MGADL_toothbrushing", "MGADL_getting_up", "MGADL_diplopia", "MGADL_ptosis",
    "MGQOL1_dissatisfaction", "MGQOL2_seeing", "MGQOL3_eating",
    "MGQOL4_social_activity_restriction", "MGQOL5_hobby_entertainment", "MGQOL6_family_role",
    "MGQOL7_behavior_modification", "MGQOL8_work_impact", "MGQOL9_speaking",
    "MGQOL10_driving", "MGQOL11_feeling_down", "MGQOL12_walking", "MGQOL13_quick_action",
    "MGQOL14_mental_crushing", "MGQOL15_dressing"
]

# ============================================================================
# ユーティリティ関数
# ============================================================================

def check_required_files():
    """必須ファイルの存在確認"""
    missing = []

    for name, path in REQUIRED_FILES.items():
        if not path.exists():
            missing.append(f"{name}: {path}")

    for model_name, paths in MODEL_FILES.items():
        for path in paths:
            if not path.exists():
                missing.append(f"{model_name}: {path}")

    if missing:
        raise FileNotFoundError(
            f"以下の必須ファイルが見つかりません:\n" + "\n".join(missing)
        )

    return True


def get_mgc_score_from_index(item_key, index):
    """MGCの選択インデックスから実際の点数を取得"""
    item = next((item for item in MGC_ITEMS if item["key"] == item_key), None)
    if item:
        return item["values"][index]
    return 0


def get_internal_key_map():
    """UIキー → 内部キー のマッピング辞書を取得"""
    key_map = {}

    # MGADL
    for item in MGADL_ITEMS:
        key_map[f"adl_{item['key']}"] = item["internal_key"]

    # MGC
    for item in MGC_ITEMS:
        key_map[f"mgc_{item['key']}"] = item["internal_key"]

    # MGQOL
    for item in MGQOL_ITEMS:
        key_map[f"mgqol_{item['key']}"] = item["internal_key"]

    return key_map
