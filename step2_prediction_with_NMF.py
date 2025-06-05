# step2. prediction with NMF matrix
import numpy as np
import pandas as pd
import seaborn as sns

import pickle
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc

seed = None

with open("./out/nmf_model.pkl", "rb") as file:
    nmf = pickle.load(file)

with open("./out/W_matrix.pkl", "rb") as file:
    W = pickle.load(file)

with open("./out/H_matrix.pkl", "rb") as file:
    H = pickle.load(file)

with open("./out/W_with_id_df.pkl", "rb") as file:
    W_with_id_df = pickle.load(file)

with open("./out/W_MM.pkl", "rb") as file:
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


def save_to_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

save_to_pickle(X_train_list, "./out/X_train_list.pkl")
save_to_pickle(y_train_list, "./out/y_train_list.pkl")
save_to_pickle(X_test_list, "./out/X_test_list.pkl")
save_to_pickle(y_test_list, "./out/y_test_list.pkl")

param_grids = {
    "svm": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf"],
        "gamma": ["auto", "scale"]
    },
    "logreg": [
        {
            "penalty": ["l1", "l2"],
            "C": [0.1, 1, 10, 100],
            "fit_intercept": [True, False],
            "solver": ["liblinear", "saga"],
        },
        {
            "penalty": ["elasticnet"],
            "C": [0.1, 1, 10, 100],
            "fit_intercept": [True, False],
            "l1_ratio": [0, 0.25, 0.5, 0.75, 1],
            "solver": ["saga"],
        },
        {
            "penalty": [None],
            "fit_intercept": [True, False],
            "solver": ["newton-cg", "lbfgs", "sag", "saga"],
        }
    ],
    "rf": {
        "n_estimators": [10, 50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }
}

# Mapping from displayed model names to the keys used in ``param_grids``
grid_key = {
    "SVM": "svm",
    "Logistic Regression": "logreg",
    "Random Forest": "rf"
}

best_params_list = []  
best_scores_list = []  
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

for X_train, y_train, X_test, y_test in zip(X_train_list, y_train_list, X_test_list, y_test_list):
    
    tuned_models = {}
    best_scores = {}
    
    for model_name, model in models.items():
        if model_name != "Naive Bayes":
            key = grid_key.get(model_name)
            assert key is not None, f"Missing grid key for {model_name}"
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[key],
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
    
    best_params = {name: model.get_params() for name, model in tuned_models.items() if name != "Naive Bayes"}
    
    best_params_list.append(best_params)
    best_scores_list.append(best_scores)
    
    for model_name, model in tuned_models.items():
        predicted_probs = model.predict_proba(X_test)[:, 1]
        predicted_probs_list[model_name].append(predicted_probs)

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


def get_optimal_cutoff(filename):
    data = pd.read_csv(filename)
    data["Youdens_Index"] = data["TPR"] - data["FPR"]
    optimal_cutoff = data.loc[data["Youdens_Index"].idxmax()]
    
    return optimal_cutoff


model_files = {
    "SVM": "./out/mean_roc_data_SVM.csv",
    "Logistic Regression": "./out/mean_roc_data_Logistic Regression.csv",
    "Random Forest": "./out/mean_roc_data_Random Forest.csv",
    "GaussianNB": "./out/mean_roc_data_Naive Bayes.csv"
}

optimal_cutoffs = {}

for model, filename in model_files.items():
    optimal_cutoff = get_optimal_cutoff(filename)
    optimal_cutoffs[model] = optimal_cutoff["Youdens_Index"]

with open("./out/optimal_cutoffs.pkl", "wb") as f:
    pickle.dump(optimal_cutoffs, f)

def save_model(model, filename):
    with open(filename, "wb") as file:
        pickle.dump(model, file)

save_model(tuned_models["SVM"], "./out/svm_model.pkl")
save_model(tuned_models["Logistic Regression"], "./out/logreg_model.pkl")
save_model(tuned_models["Random Forest"], "./out/rf_model.pkl")
save_model(tuned_models["Naive Bayes"], "./out/nb_model.pkl")


predicted_probs_files = {
    "svm": predicted_probs_list["SVM"],
    "logreg": predicted_probs_list["Logistic Regression"],
    "rf": predicted_probs_list["Random Forest"],
    "nb": predicted_probs_list["Naive Bayes"]
}

for model_name, probs_list in predicted_probs_files.items():
    save_to_pickle(probs_list, f"./out/predicted_probs_{model_name}_list.pkl")

