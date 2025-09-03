#!/usr/bin/env python
# coding: utf-8

# step2. prediction with NMF matrix

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import KFold, GridSearchCV
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score

seed = None

def save_to_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_from_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def plot_roc_curve(y_test_list, model_probs_list, model_name, save_path=None):
    plt.figure(figsize=(10, 10))
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)

    for i, (y_test, probas) in enumerate(zip(y_test_list, model_probs_list)):
        fpr, tpr, _ = roc_curve(y_test, probas)
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        plt.plot(fpr, tpr, lw=0.5, alpha=0.3, label=f'ROC fold {i+1} (AUC = {auc(fpr, tpr):.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', lw=0.5, color='black', alpha=0.5)
    mean_tpr /= len(y_test_list)
    plt.plot(mean_fpr, mean_tpr, color='blue', lw=1, alpha=0.8, label=f'Mean ROC (AUC = {auc(mean_fpr, mean_tpr):.2f})')

    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()

def save_mean_roc_data(y_test_list, model_probs_list, model_name, roc_data_dict):
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = 0.0

    for y_test, probas in zip(y_test_list, model_probs_list):
        fpr, tpr, thresholds = roc_curve(y_test, probas)
        mean_tpr += np.interp(mean_fpr, fpr, tpr)

    mean_tpr /= len(y_test_list)
    mean_auc = auc(mean_fpr, mean_tpr)

    roc_data_dict[model_name] = {
        "mean_fpr": mean_fpr,
        "mean_tpr": mean_tpr,
        "mean_auc": mean_auc
    }

def get_optimal_cutoff(filename):
    data = pd.read_csv(filename)
    data['Youdens_Index'] = data['TPR'] - data['FPR']
    optimal_cutoff = data.loc[data['Youdens_Index'].idxmax()]
    return optimal_cutoff

def calculate_and_accumulate_shap(model, X_train, X_test, model_name, shap_values_accumulated):
    explainer = shap.Explainer(model.predict_proba, X_train)
    shap_values = explainer(X_test)
    shap_values_accumulated[model_name].append(shap_values.values[:, :, 0])
    return shap_values.data

def plot_average_shap(shap_values_list, feature_data, model_name, feature_names=None):
    shap_values_mean = np.mean(np.array(shap_values_list), axis=0)
    print(f"SHAP Plot for {model_name}")
    shap.summary_plot(shap_values_mean, feature_data, feature_names=feature_names)
    plt.show()

def save_all_fold_models(all_models, base_path='./out/'):
    """全foldのモデルを保存"""
    for model_name, models_list in all_models.items():
        for fold_idx, model in enumerate(models_list):
            filename = f"{base_path}{model_name.lower().replace(' ', '_')}_fold_{fold_idx}.pkl"
            with open(filename, 'wb') as file:
                pickle.dump(model, file)

    with open(f"{base_path}all_fold_models_info.pkl", 'wb') as file:
        model_info = {
            'model_names': list(all_models.keys()),
            'num_folds': len(next(iter(all_models.values())))
        }
        pickle.dump(model_info, file)

    print(f"全foldモデル保存完了:")
    for model_name, models_list in all_models.items():
        print(f"  {model_name}: {len(models_list)}個のfoldモデル")

def main():
    with open('./out/nmf_model.pkl', 'rb') as file:
        nmf = pickle.load(file)

    with open('./out/W_matrix.pkl', 'rb') as file:
        W = pickle.load(file)

    with open('./out/H_matrix.pkl', 'rb') as file:
        H = pickle.load(file)

    with open('./out/W_with_id_df.pkl', 'rb') as file:
        W_with_id_df = pickle.load(file)

    with open('./out/W_MM.pkl', 'rb') as file:
        W_MM = pickle.load(file)

    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []

    kf = KFold(n_splits=5, random_state=seed, shuffle=True)

    for train_index, test_index in kf.split(W_MM):
        W_train, W_test = W_MM.iloc[train_index], W_MM.iloc[test_index]

        X_train = W_train[["module QOL", "module Diplopia", "module Ptosis", "module Systemic"]]
        y_train = W_train["MMorbetter"]

        X_test = W_test[["module QOL", "module Diplopia", "module Ptosis", "module Systemic"]]
        y_test = W_test["MMorbetter"]

        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_test_list.append(X_test)
        y_test_list.append(y_test)

    save_to_pickle(X_train_list, './out/X_train_list.pkl')
    save_to_pickle(y_train_list, './out/y_train_list.pkl')
    save_to_pickle(X_test_list, './out/X_test_list.pkl')
    save_to_pickle(y_test_list, './out/y_test_list.pkl')

    param_grids = {
        'svm': {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf"],
            "gamma": ["auto", "scale"]
        },
        'logreg': [
            {
                "penalty": ["l1", "l2"],
                "C": [0.1, 1, 10, 100],
                "fit_intercept": [True, False],
                "solver": ['liblinear', 'saga'],
            },
            {
                "penalty": ["elasticnet"],
                "C": [0.1, 1, 10, 100],
                "fit_intercept": [True, False],
                "l1_ratio": [0, 0.25, 0.5, 0.75, 1],
                "solver": ['saga'],
            },
            {
                "penalty": [None],
                "fit_intercept": [True, False],
                "solver": ['newton-cg', 'lbfgs', 'sag', 'saga'],
            }
        ],
        'rf': {
            "n_estimators": [10, 50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    }

    all_fold_models = {
        "SVM": [],
        "Logistic Regression": [],
        "Random Forest": [],
        "Naive Bayes": []
    }

    best_params_list = []
    best_scores_list = []

    train_predicted_probs_list = {
        "SVM": [],
        "Logistic Regression": [],
        "Random Forest": [],
        "Naive Bayes": []
    }

    predicted_probs_list = {
        "SVM": [],
        "Logistic Regression": [],
        "Random Forest": [],
        "Naive Bayes": []
    }

    models = {
        "SVM": svm.SVC(probability=True, random_state=seed),
        "Logistic Regression": LogisticRegression(random_state=seed),
        "Random Forest": RandomForestClassifier(random_state=seed),
        "Naive Bayes": GaussianNB()
    }

    for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(zip(X_train_list, y_train_list, X_test_list, y_test_list)):
        tuned_models = {}
        best_scores = {}

        for model_name, model in models.items():
            if model_name != "Naive Bayes":
                param_grid_key = {
                    "SVM": "svm",
                    "Logistic Regression": "logreg",
                    "Random Forest": "rf"
                }.get(model_name, model_name.lower())
                
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grids.get(param_grid_key, {}),
                    cv=5,
                    scoring="accuracy",
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                tuned_models[model_name] = grid_search.best_estimator_
                best_scores[model_name] = grid_search.best_score_
            else:
                model.fit(X_train, y_train)
                tuned_models[model_name] = model
                best_scores[model_name] = model.score(X_train, y_train)

        for model_name, model in tuned_models.items():
            all_fold_models[model_name].append(model)
            train_probs = model.predict_proba(X_train)[:, 1]
            train_predicted_probs_list[model_name].append(train_probs)

            test_probs = model.predict_proba(X_test)[:, 1]
            predicted_probs_list[model_name].append(test_probs)

        best_params = {name: model.get_params() for name, model in tuned_models.items() if name != "Naive Bayes"}
        best_params_list.append(best_params)
        best_scores_list.append(best_scores)

    print(f"訓練完了: 各モデルタイプで{len(all_fold_models['SVM'])}個のfoldモデルを保存")

    plot_roc_curve(
        y_test_list=y_train_list,
        model_probs_list=train_predicted_probs_list["SVM"],
        model_name="SVM (Train)",
        save_path="./fig/train_SVM.pdf"
    )

    plot_roc_curve(
        y_test_list=y_train_list,
        model_probs_list=train_predicted_probs_list["Logistic Regression"],
        model_name="Logistic Regression (Train)",
        save_path="./fig/train_logreg.pdf"
    )

    plot_roc_curve(
        y_test_list=y_train_list,
        model_probs_list=train_predicted_probs_list["Random Forest"],
        model_name="Random Forest (Train)",
        save_path="./fig/train_rf.pdf"
    )

    plot_roc_curve(
        y_test_list=y_train_list,
        model_probs_list=train_predicted_probs_list["Naive Bayes"],
        model_name="Naive Bayes (Train)",
        save_path="./fig/train_NB.pdf"
    )

    plot_roc_curve(y_test_list, predicted_probs_list["SVM"], "SVM", save_path='./fig/test_SVM.pdf')
    plot_roc_curve(y_test_list, predicted_probs_list["Logistic Regression"], "Logistic Regression", save_path='./fig/test_logreg.pdf')
    plot_roc_curve(y_test_list,  predicted_probs_list["Random Forest"], "Random Forest", save_path='./fig/test_rf.pdf')
    plot_roc_curve(y_test_list,  predicted_probs_list["Naive Bayes"], "Naive Bayes", save_path='./fig/test_NB.pdf')

    shap_values_accumulated = {
        "SVM": [],
        "Logistic Regression": [],
        "Random Forest": [],
        "Naive Bayes": []
    }

    feature_data = None
    for model_name, model in tuned_models.items():
        if hasattr(model, "predict_proba"):
            feature_data = calculate_and_accumulate_shap(model, X_train, X_test, model_name, shap_values_accumulated)

    for model_name, shap_values_list in shap_values_accumulated.items():
        if shap_values_list:
            plot_average_shap(
                shap_values_list,
                feature_data,
                model_name,
                feature_names=X_train.columns if hasattr(X_train, 'columns') else None
            )

    for model_name, shap_values_list in shap_values_accumulated.items():
        if shap_values_list:
            mean_shap_values = np.mean([np.abs(shap_vals) for shap_vals in shap_values_list], axis=0)

            shap_values = shap.Explanation(
                values=mean_shap_values,
                data=X_test,
                feature_names=X_train.columns if hasattr(X_train, 'columns') else None
            )

            print(f"SHAP Bar Plot for {model_name}")
            shap.plots.bar(shap_values)
            plt.show()

    roc_data = {}

    save_mean_roc_data(y_test_list, predicted_probs_list["SVM"], "SVM", roc_data)
    save_mean_roc_data(y_test_list, predicted_probs_list["Logistic Regression"], "Logistic Regression", roc_data)
    save_mean_roc_data(y_test_list, predicted_probs_list["Random Forest"], "Random Forest", roc_data)
    save_mean_roc_data(y_test_list, predicted_probs_list["Naive Bayes"], "Naive Bayes", roc_data)

    for model_name, data in roc_data.items():
        df = pd.DataFrame({
            "FPR": data["mean_fpr"],
            "TPR": data["mean_tpr"]
        })
        csv_path = f"./out/mean_roc_data_{model_name}.csv"
        df.to_csv(csv_path, index=False)

    model_files = {
        "SVM": './out/mean_roc_data_SVM.csv',
        "Logistic Regression": './out/mean_roc_data_Logistic Regression.csv',
        "Random Forest": './out/mean_roc_data_Random Forest.csv',
        "GaussianNB": './out/mean_roc_data_Naive Bayes.csv'
    }

    optimal_cutoffs = {}

    for model, filename in model_files.items():
        optimal_cutoff = get_optimal_cutoff(filename)
        print(f"Optimal Cutoff for {model}:")
        print(optimal_cutoff)
        print("---------------")

        optimal_cutoffs[model] = optimal_cutoff['Youdens_Index']

    with open('./out/optimal_cutoffs.pkl', 'wb') as f:
        pickle.dump(optimal_cutoffs, f)

    plt.figure(figsize=(10, 10))

    handles, labels = [], []

    for model_name, data in roc_data.items():
        mean_fpr = data["mean_fpr"]
        mean_tpr = data["mean_tpr"]
        mean_auc = data["mean_auc"]

        line, = plt.plot(mean_fpr, mean_tpr, label=f'{model_name} (AUC = {mean_auc:.2f})')
        handles.append(line)
        labels.append(f'{model_name} (AUC = {mean_auc:.2f})')

        optimal_cutoff = get_optimal_cutoff(f'./out/mean_roc_data_{model_name}.csv')
        cutoff_fpr = optimal_cutoff['FPR']
        cutoff_tpr = optimal_cutoff['TPR']

        point = plt.scatter(cutoff_fpr, cutoff_tpr, color='black', s=50, marker='x')

    handles.append(point)
    labels.append("Cutoff")

    plt.xlim([0, 1]) 
    plt.plot([1, 0], [0, 1], linestyle='--', lw=1, color='black', alpha=0)
    plt.title('')
    plt.legend(handles, labels, loc="center right")
    plt.xlabel('1-Specificity', fontsize=20)  
    plt.ylabel('Sensitivity', fontsize=20) 
    plt.title('', fontsize=18)
    plt.legend(handles, labels, loc="center right", fontsize=16)  

    plt.savefig('./fig/fig4.pdf', format='pdf', dpi=1500, transparent=False)
    plt.savefig('./fig/fig4.tiff', format='tiff', dpi=1500, transparent=False)

    plt.show()

    save_all_fold_models(all_fold_models)

    predicted_probs_files = {
        'svm': predicted_probs_list["SVM"],
        'logreg': predicted_probs_list["Logistic Regression"],
        'rf': predicted_probs_list['Random Forest'],
        'nb': predicted_probs_list['Naive Bayes']
    }

    for model_name, probs_list in predicted_probs_files.items():
        save_to_pickle(probs_list, f'./out/predicted_probs_{model_name}_list.pkl')

    print("Step 2 completed successfully!")

if __name__ == "__main__":
    main()