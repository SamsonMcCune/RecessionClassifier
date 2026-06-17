from pathlib import Path
import json
import warnings
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, classification_report, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
models = ["random_forest", "ensemble"]
project_dir = Path(__file__).parent
data_dir = project_dir / "data"
model = "ensemble"
output_dir = project_dir / f"outputs_{model}_recession"
fred_md_path = data_dir / "fred_md_current.csv"
usrec_path = data_dir / "USREC.csv"
usrec_fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC"
start_date = "1960-01-01"
test_date = "2000-01-01"
forecast_horizon = 12
feature_lag = 1
valid_years = 10
max_missing = 0.35
n_estimators = 700
max_depth = 5
min_samples_leaf = 5
seed = 42
stacking_splits = 5
model_selection_metric = "average_precision"
np.random.seed(seed)

def validate_config():
    if model not in models:
        raise ValueError(f"model must be one of {models}; got {model!r}.")
    if model_selection_metric not in ["average_precision", "roc_auc", "brier_score", "accuracy", "precision", "recall", "f1"]:
        raise ValueError("model_selection_metric must be one of: average_precision, roc_auc, brier_score, accuracy, precision, recall, f1.")

def make_run_config():
    return {"fred_md_path": str(fred_md_path), "usrec_path": str(usrec_path), "usrec_fred_url": usrec_fred_url, "start_date": start_date, "test_start": test_date, "forecast_horizon": forecast_horizon, "feature_lag": feature_lag, "valid_years": valid_years, "max_missing": max_missing, "n_estimators": n_estimators, "max_depth": max_depth, "min_samples_leaf": min_samples_leaf, "stacking_splits": stacking_splits, "model_selection_metric": model_selection_metric, "model": model, "random_state": seed, "output_dir": str(output_dir)}

def load_fred_md_from_csv(path):
    raw = pd.read_csv(Path(path).expanduser())
    raw = raw.rename(columns={raw.columns[0]: "date"})
    transform_mask = raw["date"].astype(str).str.contains("Transform", case=False, na=False)
    if not transform_mask.any():
        raise ValueError("No FRED-MD transformation row found. The standard FRED-MD file should include a row whose first column contains 'Transform'.")
    transform_codes = raw.loc[transform_mask].drop(columns="date").iloc[0].apply(pd.to_numeric, errors="coerce")
    data = raw.loc[~transform_mask].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).set_index("date").sort_index()
    data = data.apply(pd.to_numeric, errors="coerce")
    return data, transform_codes

def apply_fred_md_transformation(series, code):
    x = pd.to_numeric(series, errors="coerce")
    if pd.isna(code):
        warnings.warn(f"Missing FRED-MD transform code for {series.name}; leaving it in levels.")
        return x
    code = int(code)
    if code == 1:
        return x
    if code == 2:
        return x.diff()
    if code == 3:
        return x.diff().diff()
    if code == 4:
        return np.log(x.where(x > 0))
    if code == 5:
        return 100 * np.log(x.where(x > 0)).diff()
    if code == 6:
        return 100 * np.log(x.where(x > 0)).diff().diff()
    if code == 7:
        return 100 * (x / x.shift(1) - 1).diff()
    warnings.warn(f"Unknown FRED-MD transform code {code} for {series.name}; leaving it in levels.")
    return x

def prepare_fred_md_features(raw_features, transform_codes):
    transformed = pd.DataFrame(index=raw_features.index)
    for col in raw_features.columns:
        transformed[col] = apply_fred_md_transformation(raw_features[col], transform_codes.get(col, np.nan))
    return transformed.replace([np.inf, -np.inf], np.nan)

def add_recession_specific_features(raw_features, prepared_features):
    features = prepared_features.copy()
    if {"GS10", "TB3MS"}.issubset(raw_features.columns):
        features["TERM_SPREAD_GS10_TB3MS"] = raw_features["GS10"] - raw_features["TB3MS"]
    if {"BAA", "AAA"}.issubset(raw_features.columns):
        features["CREDIT_SPREAD_BAA_AAA"] = raw_features["BAA"] - raw_features["AAA"]
    if "UNRATE" in raw_features.columns:
        features["UNRATE_3M_CHANGE"] = raw_features["UNRATE"] - raw_features["UNRATE"].shift(3)
        features["UNRATE_12M_CHANGE"] = raw_features["UNRATE"] - raw_features["UNRATE"].shift(12)
    if "PERMIT" in raw_features.columns:
        features["PERMIT_12M_LOG_GROWTH"] = 100 * np.log(raw_features["PERMIT"].where(raw_features["PERMIT"] > 0)).diff(12)
    return features.replace([np.inf, -np.inf], np.nan)

def load_usrec_from_csv_or_fred(local_path, url=usrec_fred_url):
    source = Path(local_path).expanduser() if Path(local_path).expanduser().exists() else url
    usrec = pd.read_csv(source)
    usrec = usrec.rename(columns={usrec.columns[0]: "date"})
    usrec.columns = ["date" if c == "date" else c.upper() for c in usrec.columns]
    if "USREC" not in usrec.columns:
        raise ValueError("The recession-label file must contain a USREC column.")
    usrec["date"] = pd.to_datetime(usrec["date"], errors="coerce")
    usrec["USREC"] = pd.to_numeric(usrec["USREC"], errors="coerce")
    return usrec.dropna(subset=["date", "USREC"]).set_index("date").sort_index()

def make_forward_recession_target(usrec, horizon):
    future_recession_flags = pd.concat([usrec["USREC"].shift(-i) for i in range(1, horizon + 1)], axis=1)
    target = future_recession_flags.max(axis=1)
    target[future_recession_flags.isna().any(axis=1)] = np.nan
    return target.rename(f"recession_next_{horizon}m")

def build_modeling_frame(features, target, feature_lag, start_date):
    lagged_features = features.shift(feature_lag)
    frame = lagged_features.join(target, how="inner")
    frame = frame.loc[pd.Timestamp(start_date):].copy()
    return frame.dropna(subset=[target.name])

def select_feature_columns(frame, target_col, max_missing):
    feature_frame = frame.drop(columns=[target_col])
    missing_rate = feature_frame.isna().mean()
    candidate_cols = missing_rate[missing_rate <= max_missing].index.tolist()
    feature_counts = feature_frame[candidate_cols].nunique(dropna=True)
    feature_cols = feature_counts[feature_counts > 1].index.tolist()
    if not feature_cols:
        raise ValueError("No usable feature columns remain after missingness and variance filters.")
    return feature_cols

def split_train_valid_test(frame, target_col, feature_cols, test_start, valid_years):
    test_start = pd.Timestamp(test_start)
    pre_test = frame.loc[frame.index < test_start]
    test = frame.loc[frame.index >= test_start]
    if pre_test.empty or test.empty:
        raise ValueError("The requested test date produced an empty pre-test or test sample.")
    valid_start = pre_test.index.max() - pd.DateOffset(years=valid_years) + pd.DateOffset(months=1)
    train = pre_test.loc[pre_test.index < valid_start]
    valid = pre_test.loc[pre_test.index >= valid_start]
    if train.empty or valid.empty or test.empty:
        raise ValueError("The requested dates produced an empty train, validation, or test sample.")
    X_train, y_train = train[feature_cols], train[target_col].astype(int)
    X_valid, y_valid = valid[feature_cols], valid[target_col].astype(int)
    X_test, y_test = test[feature_cols], test[target_col].astype(int)
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def prepare_modeling_data():
    raw_features, transform_codes = load_fred_md_from_csv(fred_md_path)
    prepared_features = prepare_fred_md_features(raw_features, transform_codes)
    features = add_recession_specific_features(raw_features, prepared_features)
    usrec = load_usrec_from_csv_or_fred(usrec_path)
    target = make_forward_recession_target(usrec, forecast_horizon)
    frame = build_modeling_frame(features, target, feature_lag, start_date)
    target_col = target.name
    feature_selection_frame = frame.loc[frame.index < pd.Timestamp(test_date)]
    feature_cols = select_feature_columns(feature_selection_frame, target_col, max_missing)
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_train_valid_test(frame, target_col, feature_cols, test_date, valid_years)
    return usrec, feature_cols, X_train, y_train, X_valid, y_valid, X_test, y_test

def get_positive_class_weight(y):
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0:
        return 1.0
    return negatives / positives

def make_pipeline(estimator):
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])

def make_random_forest_pipeline():
    return make_pipeline(RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features="sqrt", class_weight="balanced_subsample", bootstrap=True, n_jobs=-1, random_state=seed))

def build_base_estimators(y_train):
    scale_pos_weight = get_positive_class_weight(y_train)
    estimators = {}
    estimators["random_forest"] = make_random_forest_pipeline()
    estimators["extra_trees"] = make_pipeline(ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features="sqrt", class_weight="balanced", bootstrap=False, n_jobs=-1, random_state=seed))
    estimators["sklearn_gradient_boosting"] = make_pipeline(GradientBoostingClassifier(n_estimators=350, learning_rate=0.03, max_depth=2, min_samples_leaf=8, subsample=0.85, random_state=seed))

    try:
        from xgboost import XGBClassifier
        estimators["xgboost"] = make_pipeline(XGBClassifier(n_estimators=450, learning_rate=0.03, max_depth=2, min_child_weight=5, subsample=0.85, colsample_bytree=0.80, objective="binary:logistic", eval_metric="logloss", tree_method="hist", scale_pos_weight=scale_pos_weight, n_jobs=-1, random_state=seed))
    except Exception as exc:
        print(f"Skipping XGBoost because it is unavailable: {type(exc).__name__}: {exc}")

    try:
        from lightgbm import LGBMClassifier
        estimators["lightgbm"] = make_pipeline(LGBMClassifier(n_estimators=450, learning_rate=0.03, max_depth=3, num_leaves=7, min_child_samples=12, subsample=0.85, colsample_bytree=0.80, class_weight="balanced", n_jobs=-1, random_state=seed, verbose=-1))
    except Exception as exc:
        print(f"Skipping LightGBM because it is unavailable: {type(exc).__name__}: {exc}")

    try:
        from catboost import CatBoostClassifier
        estimators["catboost"] = make_pipeline(CatBoostClassifier(iterations=450, learning_rate=0.03, depth=3, loss_function="Logloss", auto_class_weights="Balanced", random_seed=seed, verbose=False, allow_writing_files=False))
    except Exception as exc:
        print(f"Skipping CatBoost because it is unavailable: {type(exc).__name__}: {exc}")
    return estimators

def fit_base_estimators(estimators, X_train, y_train):
    fitted = {}
    for name, model in estimators.items():
        try:
            fitted[name] = clone(model).fit(X_train, y_train)
            print(f"Fitted base model: {name}")
        except Exception as exc:
            print(f"Skipping {name} after fit failure: {type(exc).__name__}: {exc}")
    if not fitted:
        raise ValueError("No base models fitted successfully.")
    return fitted

def predict_positive_probability(model, X):
    probabilities = model.predict_proba(X)
    classes = list(model.classes_) if hasattr(model, "classes_") else list(model.named_steps["model"].classes_)
    if 1 not in classes:
        return np.zeros(len(X), dtype=float)
    positive_index = classes.index(1)
    return probabilities[:, positive_index]

def collect_probability_matrix(fitted_models, X):
    probability_data = {name: predict_positive_probability(model, X) for name, model in fitted_models.items()}
    return pd.DataFrame(probability_data, index=X.index)

def make_time_series_oof_probabilities(X, y, n_splits):
    if len(X) < 3:
        raise ValueError("At least 3 training rows are needed for time-series stacking.")
    max_splits = min(n_splits, max(2, len(X) // 60), len(X) - 1)
    if max_splits < 2:
        raise ValueError("Not enough rows to create at least two time-series stacking splits.")
    splitter = TimeSeriesSplit(n_splits=max_splits)
    oof_parts = []
    for split_number, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
        X_fold_train, y_fold_train = X.iloc[train_idx], y.iloc[train_idx]
        X_fold_valid = X.iloc[valid_idx]
        fold_models = fit_base_estimators(build_base_estimators(y_fold_train), X_fold_train, y_fold_train)
        fold_probs = collect_probability_matrix(fold_models, X_fold_valid)
        oof_parts.append(fold_probs)
        print(f"Built stacking fold {split_number} of {max_splits}")
    oof = pd.concat(oof_parts).sort_index().dropna(axis=1, how="any").dropna(axis=0, how="any")
    if oof.empty or len(oof.columns) == 0:
        raise ValueError("No complete out-of-fold probability columns were available for stacking.")
    return oof

def fit_time_series_stacker(X_train, y_train, n_splits):
    oof_probs = make_time_series_oof_probabilities(X_train, y_train, n_splits)
    y_for_stacker = y_train.loc[oof_probs.index]
    if len(np.unique(y_for_stacker)) < 2:
        raise ValueError("The stacking target contains only one class after time-series splitting.")
    stacker = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    stacker.fit(oof_probs, y_for_stacker)
    return stacker, list(oof_probs.columns)

def make_probability_candidates(base_probability_matrix, stacker=None, stacker_columns=None):
    candidates = {col: base_probability_matrix[col].to_numpy() for col in base_probability_matrix.columns}
    candidates["soft_voting_average"] = base_probability_matrix.mean(axis=1).to_numpy()
    if stacker is not None and stacker_columns is not None:
        missing_columns = [col for col in stacker_columns if col not in base_probability_matrix.columns]
        if missing_columns:
            warnings.warn(f"Skipping stacked_logistic_ensemble because final base probabilities are missing these stacker columns: {missing_columns}")
        else:
            candidates["stacked_logistic_ensemble"] = stacker.predict_proba(base_probability_matrix[stacker_columns])[:, 1]
    return candidates

def choose_threshold_by_validation_f1(y_valid, probabilities):
    if len(np.unique(y_valid)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_valid, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1_scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.nanargmax(f1_scores)])

def evaluate_classifier(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    metrics = {"threshold": threshold, "brier_score": brier_score_loss(y_true, probabilities), "accuracy": accuracy_score(y_true, predictions), "precision": precision_score(y_true, predictions, zero_division=0), "recall": recall_score(y_true, predictions, zero_division=0), "f1": f1_score(y_true, predictions, zero_division=0)}
    metrics["roc_auc"] = roc_auc_score(y_true, probabilities) if len(np.unique(y_true)) > 1 else np.nan
    metrics["average_precision"] = average_precision_score(y_true, probabilities) if len(np.unique(y_true)) > 1 else np.nan
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    report = classification_report(y_true, predictions, labels=[0, 1], output_dict=True, zero_division=0)
    return metrics, matrix, report

def score_for_selection(metrics, selection_metric):
    if selection_metric == "brier_score":
        return -metrics["brier_score"]
    score = metrics.get(selection_metric, np.nan)
    score = metrics.get("average_precision", np.nan) if pd.isna(score) else score
    score = metrics.get("f1", -np.inf) if pd.isna(score) else score
    return score

def choose_best_candidate(y_valid, probability_candidates, selection_metric):
    rows = []
    best_name, best_threshold, best_score = None, 0.5, -np.inf
    for name, probabilities in probability_candidates.items():
        threshold = choose_threshold_by_validation_f1(y_valid, probabilities)
        metrics, _, _ = evaluate_classifier(y_valid, probabilities, threshold)
        selection_score = score_for_selection(metrics, selection_metric)
        rows.append({"candidate": name, "selection_score": selection_score, **metrics})
        if selection_score > best_score:
            best_name, best_threshold, best_score = name, threshold, selection_score
    if not rows:
        raise ValueError("No probability candidates were available for validation scoring.")
    validation_scores = pd.DataFrame(rows).sort_values("selection_score", ascending=False, na_position="last")
    return best_name, best_threshold, validation_scores

def fallback_to_available_candidate(best_candidate, threshold, validation_scores, test_candidates):
    if best_candidate in test_candidates:
        return best_candidate, threshold
    available_scores = validation_scores[validation_scores["candidate"].isin(test_candidates)]
    if available_scores.empty:
        raise ValueError("None of the validation candidates are available for the test set.")
    fallback_row = available_scores.iloc[0]
    fallback_candidate = fallback_row["candidate"]
    fallback_threshold = float(fallback_row["threshold"])
    warnings.warn(f"Best validation candidate {best_candidate!r} was unavailable at test time. Using {fallback_candidate!r} instead.")
    return fallback_candidate, fallback_threshold

def get_average_feature_importance(fitted_models, feature_cols):
    importance_frames = []
    for name, model in fitted_models.items():
        inner_model = model.named_steps["model"]
        if hasattr(inner_model, "feature_importances_"):
            values = np.asarray(inner_model.feature_importances_, dtype=float)
            if len(values) != len(feature_cols):
                warnings.warn(f"Skipping feature importances for {name}: expected {len(feature_cols)}, got {len(values)}.")
                continue
            total = values.sum()
            values = values / total if total > 0 else values
            importance_frames.append(pd.DataFrame({"feature": feature_cols, "model": name, "importance": values}))
    if not importance_frames:
        return pd.DataFrame({"feature": feature_cols, "importance": np.nan})
    combined = pd.concat(importance_frames, ignore_index=True)
    return combined.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False)


def make_probability_table(y_test, probabilities, threshold, candidate_name):
    table = pd.DataFrame(index=y_test.index)
    table["actual_target"] = y_test.values
    table["recession_probability"] = probabilities
    table["predicted_recession"] = (probabilities >= threshold).astype(int)
    table["selected_candidate"] = candidate_name
    return table

def plot_probabilities(probability_table, usrec, output_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    usrec_test = usrec.reindex(probability_table.index).ffill()
    model_title = model.replace("_", " ").title()
    ax.plot(probability_table.index, probability_table["recession_probability"], label="Predicted recession probability")
    ax.fill_between(probability_table.index, 0, 1, where=usrec_test["USREC"].astype(bool).to_numpy(), alpha=0.15, label="USREC recession month")
    ax.set_ylim(0, 1)
    ax.set_title(f"{model_title} Recession Probability")
    ax.set_ylabel("Probability")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_feature_importance(feature_importance, output_path, top_n=25):
    top = feature_importance.head(top_n).sort_values("importance")
    model_title = model.replace("_", " ").title()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["importance"])
    ax.set_title(f"Top {top_n} {model_title} Feature Importances")
    ax.set_xlabel("Average normalized importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_recession_model_dashboard(test_dates, y_test, test_probabilities, threshold, usrec, output_path):
    dashboard = pd.DataFrame({"date": pd.to_datetime(test_dates), "actual": y_test.values if hasattr(y_test, "values") else y_test, "probability": np.asarray(test_probabilities)})
    dashboard["prediction"] = (dashboard["probability"] >= threshold).astype(int)
    dashboard["false_positive"] = (dashboard["prediction"] == 1) & (dashboard["actual"] == 0)
    dashboard["false_negative"] = (dashboard["prediction"] == 0) & (dashboard["actual"] == 1)
    usrec_plot = usrec.copy().reset_index()
    usrec_plot["date"] = pd.to_datetime(usrec_plot["date"])
    usrec_plot = usrec_plot[(usrec_plot["date"] >= dashboard["date"].min()) & (usrec_plot["date"] <= dashboard["date"].max())]
    model_title = model.replace("_", " ").title()
    plt.figure(figsize=(14, 7))
    for _, row in usrec_plot[usrec_plot["USREC"] == 1].iterrows():
        plt.axvspan(row["date"], row["date"] + pd.DateOffset(months=1), alpha=0.15)
    plt.plot(dashboard["date"], dashboard["probability"], label="Predicted recession probability", linewidth=2)
    plt.axhline(threshold, linestyle="--", label=f"Classification threshold = {threshold:.2f}")
    fp = dashboard[dashboard["false_positive"]]
    fn = dashboard[dashboard["false_negative"]]
    plt.scatter(fp["date"], fp["probability"], marker="x", s=60, label="False positives")
    plt.scatter(fn["date"], fn["probability"], marker="o", s=60, label="False negatives")
    plt.title(f"{model_title} Recession Classifier Dashboard")
    plt.ylabel("Predicted probability")
    plt.xlabel("Date")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def save_outputs(output_dir, model_bundle, metrics, matrix, report, probability_table, feature_importance, validation_scores, run_config, usrec):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, output_dir / f"{model}_recession_classifier.joblib")
    pd.DataFrame([metrics]).to_csv(output_dir / "test_metrics.csv", index=False)
    pd.DataFrame(matrix, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]).to_csv(output_dir / "confusion_matrix.csv")
    pd.DataFrame(report).T.to_csv(output_dir / "classification_report.csv")
    probability_table.to_csv(output_dir / "test_recession_probabilities.csv")
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    validation_scores.to_csv(output_dir / "validation_candidate_scores.csv", index=False)
    plot_probabilities(probability_table, usrec, output_dir / "recession_probabilities.png")
    plot_feature_importance(feature_importance, output_dir / "feature_importance.png")
    plot_recession_model_dashboard(probability_table.index, probability_table["actual_target"], probability_table["recession_probability"], metrics["threshold"], usrec, output_dir / "recession_model_dashboard.png")
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

def print_run_summary(metrics, selected_candidate, training_rows, test_rows, feature_count):
    print("Test metrics")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, (int, float, np.floating)) else f"{key}: {value}")
    print(f"\nSelected validation candidate: {selected_candidate}")
    print(f"Saved outputs to: {output_dir.resolve()}")
    print(f"Training rows: {training_rows}, test rows: {test_rows}, features: {feature_count}")

def random_forest():
    run_config = make_run_config()
    usrec, feature_cols, X_train, y_train, X_valid, y_valid, X_test, y_test = prepare_modeling_data()
    train_model = clone(make_random_forest_pipeline()).fit(X_train, y_train)
    valid_probabilities = predict_positive_probability(train_model, X_valid)
    valid_candidates = {"random_forest": valid_probabilities}
    best_candidate, threshold, validation_scores = choose_best_candidate(y_valid, valid_candidates, model_selection_metric)

    X_train_final = pd.concat([X_train, X_valid])
    y_train_final = pd.concat([y_train, y_valid])
    final_model = clone(make_random_forest_pipeline()).fit(X_train_final, y_train_final)
    test_probabilities = predict_positive_probability(final_model, X_test)

    metrics, matrix, report = evaluate_classifier(y_test, test_probabilities, threshold)
    metrics["selected_candidate"] = best_candidate
    metrics["model"] = model
    probability_table = make_probability_table(y_test, test_probabilities, threshold, best_candidate)
    feature_importance = get_average_feature_importance({"random_forest": final_model}, feature_cols)
    model_bundle = {"model": model, "selected_candidate": best_candidate, "threshold": threshold, "model_object": final_model, "feature_cols": feature_cols, "run_config": run_config}
    save_outputs(output_dir, model_bundle, metrics, matrix, report, probability_table, feature_importance, validation_scores, run_config, usrec)
    print_run_summary(metrics, best_candidate, len(X_train_final), len(X_test), len(feature_cols))

def ensemble():
    run_config = make_run_config()
    usrec, feature_cols, X_train, y_train, X_valid, y_valid, X_test, y_test = prepare_modeling_data()
    base_estimators = build_base_estimators(y_train)
    fitted_train_models = fit_base_estimators(base_estimators, X_train, y_train)
    try:
        stacker, stacker_columns = fit_time_series_stacker(X_train, y_train, stacking_splits)
    except Exception as exc:
        print(f"Skipping stacked logistic ensemble: {type(exc).__name__}: {exc}")
        stacker, stacker_columns = None, None
    valid_base_probs = collect_probability_matrix(fitted_train_models, X_valid)
    valid_candidates = make_probability_candidates(valid_base_probs, stacker, stacker_columns)
    best_candidate, threshold, validation_scores = choose_best_candidate(y_valid, valid_candidates, model_selection_metric)

    X_train_final = pd.concat([X_train, X_valid])
    y_train_final = pd.concat([y_train, y_valid])
    fitted_final_models = fit_base_estimators(build_base_estimators(y_train_final), X_train_final, y_train_final)
    try:
        final_stacker, final_stacker_columns = fit_time_series_stacker(X_train_final, y_train_final, stacking_splits)
    except Exception as exc:
        print(f"Skipping final stacked logistic ensemble: {type(exc).__name__}: {exc}")
        final_stacker, final_stacker_columns = None, None
    test_base_probs = collect_probability_matrix(fitted_final_models, X_test)
    test_candidates = make_probability_candidates(test_base_probs, final_stacker, final_stacker_columns)
    best_candidate, threshold = fallback_to_available_candidate(best_candidate, threshold, validation_scores, test_candidates)
    test_probabilities = test_candidates[best_candidate]

    metrics, matrix, report = evaluate_classifier(y_test, test_probabilities, threshold)
    metrics["selected_candidate"] = best_candidate
    metrics["model"] = model
    probability_table = make_probability_table(y_test, test_probabilities, threshold, best_candidate)
    feature_importance = get_average_feature_importance(fitted_final_models, feature_cols)
    model_bundle = {"model": model, "selected_candidate": best_candidate, "threshold": threshold, "base_models": fitted_final_models, "stacker": final_stacker, "stacker_columns": final_stacker_columns, "feature_cols": feature_cols, "run_config": run_config}
    save_outputs(output_dir, model_bundle, metrics, matrix, report, probability_table, feature_importance, validation_scores, run_config, usrec)
    print_run_summary(metrics, best_candidate, len(X_train_final), len(X_test), len(feature_cols))

def run():
    validate_config()
    if model == "random_forest":
        random_forest()
    elif model == "ensemble":
        ensemble()

if __name__ == "__main__":
    run()
