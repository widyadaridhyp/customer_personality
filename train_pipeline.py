import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except:
    _HAS_XGB = False

import optuna
from optuna.samplers import TPESampler

from app.preprocessing import preprocessing_raw


# ===========================================================
# Directories
# ===========================================================
MODEL_DIR = Path("model")
DATA_PATH = Path("data/data_fix.csv")
MODEL_DIR.mkdir(exist_ok=True)


# ===========================================================
# Preprocessor Builder
# ===========================================================
def build_preprocessor(minmax_cols, pass_cols):
    return ColumnTransformer([
        ("minmax", MinMaxScaler(), minmax_cols),
        ("pass", "passthrough", pass_cols)
    ])


# ===========================================================
# Main Training Pipeline
# ===========================================================
def main():

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    df = preprocessing_raw(df)

    y = df["response"]
    X = df.drop(columns=["response"])

    # Feature sets
    minmax_cols = [
        "income","recency","numwebvisitsmonth","numwebpurchases",
        "numstorepurchases","monetary","numdealspurchases","numcatalogpurchases"
    ]
    pass_cols = ["education_ord", "marital_status_ord"]

    features = [c for c in (minmax_cols + pass_cols) if c in X.columns]

    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X[features], y, test_size=0.3, random_state=42, stratify=y
    )

    # Underfitting
    rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

    # Preprocessing + PCA
    preprocessor = build_preprocessor(minmax_cols, pass_cols)

    base_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("pca", PCA(n_components=0.95))
    ])

    X_train_trans = base_pipeline.fit_transform(X_train_res)
    X_test_trans = base_pipeline.transform(X_test)

    # ===========================================================
    # Model Dictionary
    # ===========================================================
    positive = sum(y_train_res==1)
    negative = sum(y_train_res==0)

    model_dict = {
        "logit": LogisticRegression(class_weight='balanced', max_iter=500),
        "knn": KNeighborsClassifier(5),
        "dt": DecisionTreeClassifier(random_state=42),
        "rf": RandomForestClassifier(class_weight='balanced', random_state=42),
        "gb": GradientBoostingClassifier(random_state=42)
    }

    if _HAS_XGB:
        model_dict["xgb"] = XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
            scale_pos_weight=negative/positive
        )

    # ===========================================================
    # OPTUNA OBJECTIVE (Optimize Recall)
    # ===========================================================
    def objective(trial):

        model_name = trial.suggest_categorical(
            "model",
            list(model_dict.keys())
        )

        # Base model
        model = model_dict[model_name]

        # --------------------
        # Hyperparameter Space
        # --------------------
        if model_name == "logit":
            C = trial.suggest_float("C", 0.001, 10, log=True)
            model = LogisticRegression(
                C=C,
                class_weight='balanced',
                max_iter=500
            )

        elif model_name == "knn":
            n = trial.suggest_int("n_neighbors", 3, 25)
            model = KNeighborsClassifier(n_neighbors=n)

        elif model_name == "dt":
            md = trial.suggest_int("max_depth", 2, 20)
            model = DecisionTreeClassifier(max_depth=md, random_state=42)

        elif model_name == "rf":
            n_est = trial.suggest_int("n_estimators", 100, 800)
            md = trial.suggest_int("max_depth", 2, 20)
            model = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=md,
                class_weight="balanced",
                random_state=42
            )

        elif model_name == "gb":
            n_est = trial.suggest_int("n_estimators", 100, 800)
            lr = trial.suggest_float("learning_rate", 0.01, 0.3)
            model = GradientBoostingClassifier(
                n_estimators=n_est,
                learning_rate=lr,
                random_state=42
            )

        elif model_name == "xgb":
            md = trial.suggest_int("max_depth", 3, 10)
            lr = trial.suggest_float("learning_rate", 0.01, 0.3)
            n_est = trial.suggest_int("n_estimators", 100, 800)
            sub = trial.suggest_float("subsample", 0.5, 1.0)
            model = XGBClassifier(
                max_depth=md,
                learning_rate=lr,
                n_estimators=n_est,
                subsample=sub,
                scale_pos_weight=negative/positive,
                eval_metric="logloss",
                random_state=42,
                use_label_encoder=False
            )

        # Fit on transformed data
        model.fit(X_train_trans, y_train_res)
        y_pred = model.predict(X_test_trans)

        # Evaluate Recall
        report = classification_report(y_test, y_pred, output_dict=True)
        recall = report["1"]["recall"]

        return recall

    # ===========================================================
    # RUN OPTUNA
    # ===========================================================
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=50)

    print("Best Recall:", study.best_value)
    print("Best Params:", study.best_params)

    best = study.best_params
    best_name = best["model"]

    # ===========================================================
    # Rebuild best model with best params
    # ===========================================================
    if best_name == "logit":
        best_model = LogisticRegression(
            C=best["C"],
            class_weight='balanced',
            max_iter=500
        )

    elif best_name == "knn":
        best_model = KNeighborsClassifier(
            n_neighbors=best["n_neighbors"]
        )

    elif best_name == "dt":
        best_model = DecisionTreeClassifier(
            max_depth=best["max_depth"],
            random_state=42
        )

    elif best_name == "rf":
        best_model = RandomForestClassifier(
            n_estimators=best["n_estimators"],
            max_depth=best["max_depth"],
            class_weight="balanced",
            random_state=42
        )

    elif best_name == "gb":
        best_model = GradientBoostingClassifier(
            n_estimators=best["n_estimators"],
            learning_rate=best["learning_rate"],
            random_state=42
        )

    elif best_name == "xgb":
        best_model = XGBClassifier(
            max_depth=best["max_depth"],
            learning_rate=best["learning_rate"],
            n_estimators=best["n_estimators"],
            subsample=best["subsample"],
            scale_pos_weight=negative/positive,
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False
        )

    # Train final best model
    best_model.fit(X_train_trans, y_train_res)

    # ===========================================================
    # Threshold tuning (BASED ON RECALL)
    # ===========================================================
    threshold = 0.5
    if hasattr(best_model, "predict_proba"):

        y_prob = best_model.predict_proba(X_test_trans)[:, 1]
        best_recall = -1

        for t in np.linspace(0.01, 0.99, 99):
            preds = (y_prob >= t).astype(int)
            rec = recall_score(y_test, preds)

            if rec > best_recall:
                best_recall = rec
                threshold = t

    # ===========================================================
    # Final Pipeline (Preprocess + PCA + Best Model)
    # ===========================================================
    final_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("pca", base_pipeline.named_steps["pca"]),
        ("model", best_model)
    ])

    final_pipeline.fit(X_train_res, y_train_res)

    # Save artifacts
    with open(MODEL_DIR / "pipeline.pkl", "wb") as f:
        pickle.dump(final_pipeline, f)

    with open(MODEL_DIR / "features.pkl", "wb") as f:
        pickle.dump(features, f)

    with open(MODEL_DIR / "threshold.pkl", "wb") as f:
        pickle.dump(threshold, f)

    print("Training done. Models saved!")


if __name__ == "__main__":
    main()
