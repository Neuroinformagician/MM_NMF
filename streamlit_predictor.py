"""
Streamlit MG Prediction App - Predictor
予測ロジック: NMF変換 + 機械学習モデルによるアンサンブル予測
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import streamlit as st

from streamlit_config import (
    REQUIRED_FILES, MODEL_FILES, DF_COLUMNS,
    MGC_ITEMS, get_mgc_score_from_index
)


class MGPredictor:
    """MG予測クラス"""

    def __init__(self):
        """初期化"""
        self.models = None
        self.H_matrix = None
        self.W_MM = None
        self.module_names = None
        self.scaler = None
        self.df_original = None
        self.optimal_cutoffs = None

    def load_resources(self):
        """リソース読み込み（キャッシュ）"""
        try:
            # H行列読み込み
            with open(REQUIRED_FILES["H_matrix"], 'rb') as f:
                self.H_matrix = pickle.load(f)

            # W_MM読み込み（MM群/non-MM群の平均スコア）
            with open(REQUIRED_FILES["W_MM"], 'rb') as f:
                self.W_MM = pickle.load(f)

            # モジュール名読み込み
            with open(REQUIRED_FILES["module_names"], 'rb') as f:
                self.module_names = pickle.load(f)

            # 元データ読み込み（スケーラー学習用）
            self.df_original = pd.read_csv(REQUIRED_FILES["original_data"], index_col=0)

            # スケーラー学習
            self.scaler = MinMaxScaler()
            self.scaler.fit(self.df_original[DF_COLUMNS])

            # モデル読み込み
            self.models = {}
            for model_name, paths in MODEL_FILES.items():
                self.models[model_name] = []
                for path in paths:
                    with open(path, 'rb') as f:
                        model = pickle.load(f)
                        self.models[model_name].append(model)

            # 最適カットオフ読み込み
            with open(REQUIRED_FILES["optimal_cutoffs"], 'rb') as f:
                cutoffs = pickle.load(f)
                # GaussianNB を Naive Bayes に対応させる
                self.optimal_cutoffs = {
                    'SVM': float(cutoffs['SVM']),
                    'Logistic Regression': float(cutoffs['Logistic Regression']),
                    'Random Forest': float(cutoffs['Random Forest']),
                    'Naive Bayes': float(cutoffs['GaussianNB'])
                }

            return True

        except Exception as e:
            st.error(f"リソース読み込みエラー: {e}")
            return False

    def convert_ui_scores_to_dataframe(self, ui_scores):
        """
        UIからの入力スコアを内部形式のDataFrameに変換

        Parameters
        ----------
        ui_scores : dict
            UIからの入力スコア
            例: {"adl_speech": 0, "mgc_ptosis": 1, "mgqol_q1": 0, ...}

        Returns
        -------
        pd.DataFrame
            内部形式のDataFrame（33カラム）
        """
        # 内部形式の辞書を作成
        internal_data = {}

        # MGC項目の変換（実際の点数に変換）
        for item in MGC_ITEMS:
            ui_key = f"mgc_{item['key']}"
            internal_key = item["internal_key"]

            if ui_key in ui_scores:
                index = ui_scores[ui_key]
                actual_score = item["values"][index]
                internal_data[internal_key] = actual_score
            else:
                internal_data[internal_key] = 0

        # MGADL項目の変換（そのまま）
        from streamlit_config import MGADL_ITEMS
        for item in MGADL_ITEMS:
            ui_key = f"adl_{item['key']}"
            internal_key = item["internal_key"]

            if ui_key in ui_scores:
                internal_data[internal_key] = ui_scores[ui_key]
            else:
                internal_data[internal_key] = 0

        # MGQOL項目の変換（そのまま）
        from streamlit_config import MGQOL_ITEMS
        for item in MGQOL_ITEMS:
            ui_key = f"mgqol_{item['key']}"
            internal_key = item["internal_key"]

            if ui_key in ui_scores:
                internal_data[internal_key] = ui_scores[ui_key]
            else:
                internal_data[internal_key] = 0

        # DataFrameに変換（カラム順序を保証）
        df = pd.DataFrame([internal_data])[DF_COLUMNS]
        return df

    def transform_to_modules(self, patient_df):
        """
        患者データをNMFでモジュールスコアに変換

        Parameters
        ----------
        patient_df : pd.DataFrame
            患者データ（33カラム）

        Returns
        -------
        pd.DataFrame
            モジュールスコア（4カラム）
        """
        # スケーリング
        patient_scaled = self.scaler.transform(patient_df)

        # 元データとconcat
        df_scaled = pd.DataFrame(
            self.scaler.transform(self.df_original[DF_COLUMNS]),
            columns=DF_COLUMNS
        )
        concatenated_df_scaled = pd.concat(
            [df_scaled, pd.DataFrame(patient_scaled, columns=DF_COLUMNS)],
            axis=0
        )

        # NMF変換（元のコードと同じロジック）
        np.random.seed(0)
        max_iter = 100

        W_concatenated = np.random.rand(
            concatenated_df_scaled.shape[0],
            self.H_matrix.shape[0]
        )

        for _ in range(max_iter):
            numer = concatenated_df_scaled.values @ self.H_matrix.T
            denom = W_concatenated @ (self.H_matrix @ self.H_matrix.T)
            denom = np.where(denom == 0, 1, denom)
            W_concatenated *= numer / denom

        # 患者のモジュールスコアを取得（最後の行）
        W_patient = W_concatenated[-1:, :]

        # DataFrameに変換
        module_df = pd.DataFrame(W_patient, columns=self.module_names)

        return module_df

    def predict(self, module_scores):
        """
        モジュールスコアから予測実行

        Parameters
        ----------
        module_scores : pd.DataFrame
            モジュールスコア（4カラム）

        Returns
        -------
        dict
            予測結果
            {
                "module_scores": pd.DataFrame,
                "predictions": {
                    "SVM": [確率1, 確率2, ...],
                    "Logistic Regression": [...],
                    "Random Forest": [...],
                    "Naive Bayes": [...]
                },
                "mean_predictions": {
                    "SVM": 平均確率,
                    ...
                },
                "ensemble": アンサンブル確率,
                "classification": "MM or better" or "non MM"
            }
        """
        # Logistic Regressionを除外
        predictions = {
            "SVM": [],
            "Random Forest": [],
            "Naive Bayes": []
        }

        # 各モデル × 各foldで予測
        for model_name in predictions.keys():
            for fold_idx in range(5):
                model = self.models[model_name][fold_idx]
                prob = model.predict_proba(module_scores)[:, 1][0]
                predictions[model_name].append(prob)

        # 各モデルの平均
        mean_predictions = {
            model_name: np.mean(probs)
            for model_name, probs in predictions.items()
        }

        # アンサンブル（全モデル×全foldの平均）
        all_probs = []
        for probs in predictions.values():
            all_probs.extend(probs)
        ensemble_prob = np.mean(all_probs)

        # Soft Voting: 各モデルの最適カットオフで判定し、多数決
        model_votes = {}
        for model_name, mean_prob in mean_predictions.items():
            cutoff = self.optimal_cutoffs[model_name]
            vote = 1 if mean_prob >= cutoff else 0  # 1: MM or better, 0: non MM
            model_votes[model_name] = {
                'probability': mean_prob,
                'cutoff': cutoff,
                'prediction': "MM or better" if vote == 1 else "non MM"
            }

        # 多数決で最終判定（3モデル中2つ以上）
        total_votes = sum(1 for v in model_votes.values() if v['prediction'] == "MM or better")
        classification = "MM or better" if total_votes >= 2 else "non MM"  # 3モデル中2つ以上

        return {
            "module_scores": module_scores,
            "predictions": predictions,
            "mean_predictions": mean_predictions,
            "model_votes": model_votes,
            "ensemble": ensemble_prob,
            "total_mm_votes": total_votes,
            "total_models": len(model_votes),
            "classification": classification
        }

    def get_mm_comparison_data(self):
        """
        MM群/non-MM群の平均スコアを取得

        Returns
        -------
        dict
            {
                "mm_avg": np.array,
                "non_mm_avg": np.array,
                "module_names": list
            }
        """
        mm_group = self.W_MM[self.W_MM["MMorbetter"] == 1][self.module_names].mean()
        non_mm_group = self.W_MM[self.W_MM["MMorbetter"] == 0][self.module_names].mean()

        return {
            "mm_avg": mm_group.values,
            "non_mm_avg": non_mm_group.values,
            "module_names": self.module_names
        }
