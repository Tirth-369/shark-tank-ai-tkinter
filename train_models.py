"""Train the six Shark Tank AI prediction tasks from the supplied historical CSV.
All preprocessing is fitted inside sklearn Pipelines to avoid train/test leakage.
Currency values in this dataset are expressed in INR lakh.
"""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score)
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
    RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data.csv"
if not DATA_PATH.exists(): DATA_PATH = ROOT / "data(2).csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
SHARKS = ["Namita", "Vineeta", "Anupam", "Aman", "Peyush"]
NUMERIC = [
    "Started in", "Number of Presenters", "Pitchers Average Age",
    "Yearly Revenue", "Monthly Sales", "Gross Margin", "Net Margin", "EBITDA",
    "Cash Burn", "SKUs", "Has Patents",
    "Original Ask Amount", "Original Offered Equity", "Valuation Requested",
    "Namita Present", "Vineeta Present", "Anupam Present", "Aman Present",
    "Peyush Present", "Ritesh Present", "Amit Present", "Guest Present",
    "ask_log", "valuation_requested_log",
    "ask_equity_ratio", "valuation_ask_ratio", "sharks_present_count"
]
CATEGORICAL = ["Industry", "Pitchers State", "Bootstrapped"]

def clean_frame(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if "Pitchers State" in df:
        df["Pitchers State"] = (df["Pitchers State"].replace("Vijayawada", "Andhra Pradesh")
            .fillna("Unknown").astype(str).str.split(",").str[0].str.strip())
    if "Bootstrapped" in df:
        df["Bootstrapped"] = df["Bootstrapped"].fillna("unknown").astype(str).str.lower()
    for col in ["Original Ask Amount", "Original Offered Equity", "Valuation Requested"]:
        if col not in df: df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ask_log"] = np.log1p(df["Original Ask Amount"].clip(lower=0))
    df["valuation_requested_log"] = np.log1p(df["Valuation Requested"].clip(lower=0))
    df["ask_equity_ratio"] = df["Original Offered Equity"] / (df["Original Ask Amount"].fillna(0)+1)
    df["valuation_ask_ratio"] = df["Valuation Requested"] / (df["Original Ask Amount"].fillna(0)+1)
    present_cols = [f"{s} Present" for s in ["Namita","Vineeta","Anupam","Aman","Peyush","Ritesh","Amit","Guest"]]
    for c in present_cols:
        if c not in df: df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["sharks_present_count"] = df[present_cols].sum(axis=1)
    if "Pitchers Average Age" in df and df["Pitchers Average Age"].dtype == object:
        df["Pitchers Average Age"] = df["Pitchers Average Age"].map({"Young":0,"Middle":1,"Old":2})
    for c in NUMERIC:
        if c not in df: df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CATEGORICAL:
        if c not in df: df[c] = "Unknown"
        df[c] = df[c].fillna("Unknown").astype(str)
    return df

def features_for(df):
    return [c for c in NUMERIC+CATEGORICAL if c in df.columns]

def make_pipeline(X, model, scale_numeric=True):
    nums = [c for c in X.columns if c in NUMERIC]
    cats = [c for c in X.columns if c in CATEGORICAL]
    prep = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                      ("scale", StandardScaler() if scale_numeric else "passthrough")]), nums),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cats)
    ], remainder="drop")
    return Pipeline([("preprocess", prep), ("model", model)])

def task_groups(raw, index):
    season = raw.loc[index, "Season Number"].astype(str)
    episode = raw.loc[index, "Episode Number"].astype(str)
    return season + "-" + episode

def development_test_indices(y, groups, classification):
    if classification:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        splitter = GroupKFold(n_splits=5, shuffle=True, random_state=42)
    development_idx, test_idx = next(splitter.split(np.zeros(len(y)), y, groups))
    return development_idx, test_idx

def development_cv_splits(y, groups, classification):
    if classification:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
    else:
        splitter = GroupKFold(n_splits=5, shuffle=True, random_state=43)
    return list(splitter.split(np.zeros(len(y)), y, groups))

def classification_candidates():
    return [
        ("LogisticRegression", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42), True),
        ("RandomForestClassifier", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            max_features="sqrt", random_state=42, n_jobs=-1), False),
        ("ExtraTreesClassifier", ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            max_features="sqrt", random_state=42, n_jobs=-1), False),
    ]

def regression_candidates():
    return [
        ("Ridge", Ridge(alpha=1.0), True),
        ("RandomForestRegressor", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, max_features=0.9,
            random_state=42, n_jobs=-1), False),
        ("ExtraTreesRegressor", ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2, max_features=0.9,
            random_state=42, n_jobs=-1), False),
    ]

def positive_probability(model, X):
    probabilities = model.predict_proba(X)
    classes = list(model.named_steps["model"].classes_)
    return probabilities[:, classes.index(1)] if 1 in classes else np.zeros(len(X))

def regression_metrics(actual, predicted, target):
    return {
        "r2": float(r2_score(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "test_rows": int(len(actual)),
        "mae_unit": "₹ lakh" if "Amount" in target or "Valuation" in target else "percentage points",
    }

def class_metrics(y, prob):
    classes = np.unique(y)
    pred = (prob >= 0.5).astype(int)
    return {"accuracy":float(accuracy_score(y,pred)),
        "precision":float(precision_score(y,pred,zero_division=0)),
        "recall":float(recall_score(y,pred,zero_division=0)),
        "f1":float(f1_score(y,pred,zero_division=0)),
        "roc_auc":float(roc_auc_score(y,prob)) if len(classes)==2 else None,
        "pr_auc":float(average_precision_score(y,prob)) if len(classes)==2 else None,
        "confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist(),
        "class_counts":{"negative":int(np.sum(np.asarray(y)==0)),
                        "positive":int(np.sum(np.asarray(y)==1))},
        "test_rows":int(len(y))}

def aggregate_fold_metrics(fold_metrics, keys):
    return {key:{"mean":float(np.mean([metrics[key] for metrics in fold_metrics])),
                 "std":float(np.std([metrics[key] for metrics in fold_metrics], ddof=1))}
            for key in keys}

def class_distribution(y):
    counts = pd.Series(y).value_counts().to_dict()
    return {"negative":int(counts.get(0, 0)), "positive":int(counts.get(1, 0))}

def regression_diagnostics(actual, predicted):
    residuals = np.asarray(actual) - np.asarray(predicted)
    diagnostics = {
        "target_min":float(np.min(actual)),
        "target_median":float(np.median(actual)),
        "target_95th_percentile":float(np.percentile(actual, 95)),
        "target_max":float(np.max(actual)),
        "residual_mean":float(np.mean(residuals)),
        "residual_std":float(np.std(residuals)),
        "max_absolute_error":float(np.max(np.abs(residuals))),
    }
    bins = pd.qcut(pd.Series(actual), q=4, duplicates="drop")
    diagnostics["target_range_errors"] = []
    for interval in bins.cat.categories:
        mask = bins == interval
        if mask.any():
            range_actual = np.asarray(actual)[mask.to_numpy()]
            range_predicted = np.asarray(predicted)[mask.to_numpy()]
            diagnostics["target_range_errors"].append({
                "range":str(interval),
                "samples":int(mask.sum()),
                "mae":float(mean_absolute_error(range_actual, range_predicted)),
                "rmse":float(np.sqrt(mean_squared_error(range_actual, range_predicted))),
            })
    return diagnostics

def model_explanation(model, input_features):
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        raw_scores = np.asarray(estimator.feature_importances_)
        method = "Tree feature importance"
    elif hasattr(estimator, "coef_"):
        raw_scores = np.mean(np.abs(np.asarray(estimator.coef_)), axis=0)
        method = "Absolute linear coefficient"
    else:
        return {"method": "Not available", "features": []}

    transformed_names = model.named_steps["preprocess"].get_feature_names_out()
    scores = {}
    for transformed_name, score in zip(transformed_names, raw_scores):
        name = transformed_name.split("__", 1)[-1]
        source = next((feature for feature in input_features
                       if name == feature or name.startswith(feature + "_")), name)
        scores[source] = scores.get(source, 0.0) + float(score)
    top_features = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:10]
    return {"method": method,
            "note": "Associations in the fitted model, not causal effects.",
            "features": [{"name": name, "importance": score} for name, score in top_features]}

def cross_validated_permutation_importance(X, y, cv_splits, estimator,
                                           scale_numeric, classification):
    fold_importances = []
    scoring = "roc_auc" if classification else "r2"
    for train_idx, validation_idx in cv_splits:
        fold_model = make_pipeline(X.iloc[train_idx], clone(estimator), scale_numeric)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        importance = permutation_importance(
            fold_model, X.iloc[validation_idx], y.iloc[validation_idx],
            scoring=scoring, n_repeats=3, random_state=42, n_jobs=1)
        fold_importances.append(importance.importances_mean)
    means = np.mean(fold_importances, axis=0)
    deviations = np.std(fold_importances, axis=0, ddof=1)
    ranked = sorted(zip(X.columns, means, deviations), key=lambda values: values[1], reverse=True)
    return {
        "method": f"Grouped-CV permutation importance ({scoring} score decrease)",
        "note": "Computed on development validation folds only. Predictive association, not causation.",
        "features": [{"name": name, "mean": float(mean), "std": float(std)}
                     for name, mean, std in ranked[:10]],
    }

def select_classifier(X, y, groups):
    development_idx, test_idx = development_test_indices(y, groups, True)
    X_development, X_test = X.iloc[development_idx], X.iloc[test_idx]
    y_development, y_test = y.iloc[development_idx], y.iloc[test_idx]
    groups_development = groups.iloc[development_idx]
    cv_splits = development_cv_splits(y_development, groups_development, True)
    comparisons = []
    for candidate_name, estimator, scale_numeric in classification_candidates():
        fold_metrics = []
        for train_idx, validation_idx in cv_splits:
            candidate = make_pipeline(X_development.iloc[train_idx], clone(estimator), scale_numeric)
            candidate.fit(X_development.iloc[train_idx], y_development.iloc[train_idx])
            probability = positive_probability(candidate, X_development.iloc[validation_idx])
            fold_metrics.append(class_metrics(y_development.iloc[validation_idx].to_numpy(), probability))
        comparisons.append({
            "model": candidate_name,
            "selection_metric": "roc_auc",
            "cv_metrics": aggregate_fold_metrics(fold_metrics,
                ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"]),
        })

    best = max(comparisons, key=lambda result: result["cv_metrics"]["roc_auc"]["mean"])
    selected_name = best["model"]
    selected_estimator, selected_scale = next(
        (estimator, scale) for name, estimator, scale in classification_candidates()
        if name == selected_name)
    selected = make_pipeline(X_development, clone(selected_estimator), selected_scale)
    selected.fit(X_development, y_development)
    explanation = model_explanation(selected, list(X.columns))
    explanation["permutation"] = cross_validated_permutation_importance(
        X_development, y_development, cv_splits, selected_estimator,
        selected_scale, True)
    test_probability = positive_probability(selected, X_test)
    test_metrics = class_metrics(y_test.to_numpy(), test_probability)

    fold_distributions = [{
        "fold": fold_number,
        "training": class_distribution(y_development.iloc[train_idx]),
        "validation": class_distribution(y_development.iloc[validation_idx]),
    } for fold_number, (train_idx, validation_idx) in enumerate(cv_splits, start=1)]
    return selected, {
        "model": selected_name,
        "selection_metric": "roc_auc",
        "comparisons": comparisons,
        "metrics": test_metrics,
        "class_distribution": {
            "development": class_distribution(y_development),
            "test": class_distribution(y_test),
            "cv_folds": fold_distributions,
        },
        "training_rows": int(len(X_development)),
        "test_groups": int(groups.iloc[test_idx].nunique()),
        "features": list(X.columns),
        "explanation": explanation,
        "split_method": "StratifiedGroupKFold by season and episode",
    }

def train_classifier(df, X_all, target, model_name, task_name, raw, subset=None):
    data = df if subset is None else df.loc[subset]
    X = X_all.loc[data.index]
    y = pd.to_numeric(data[target], errors="coerce").fillna(0).astype(int)
    if y.nunique() < 2: raise ValueError(f"{task_name}: target has fewer than two classes")
    groups = task_groups(raw, X.index)
    model, result = select_classifier(X, y, groups)
    joblib.dump(model, MODEL_DIR/f"{model_name}.pkl")
    result.update({"target":target,"target_type":"classification"})
    return result

def select_regressor(X, y_log, y_original, groups, target):
    development_idx, test_idx = development_test_indices(y_log, groups, False)
    X_development, X_test = X.iloc[development_idx], X.iloc[test_idx]
    y_development = y_log.iloc[development_idx]
    y_test_original = y_original.iloc[test_idx].to_numpy()
    groups_development = groups.iloc[development_idx]
    cv_splits = development_cv_splits(y_development, groups_development, False)
    comparisons = []
    for candidate_name, estimator, scale_numeric in regression_candidates():
        fold_actual = []
        fold_predicted = []
        for train_idx, validation_idx in cv_splits:
            candidate = make_pipeline(X_development.iloc[train_idx], clone(estimator), scale_numeric)
            candidate.fit(X_development.iloc[train_idx], y_development.iloc[train_idx])
            validation_log = candidate.predict(X_development.iloc[validation_idx])
            fold_actual.append(np.expm1(y_development.iloc[validation_idx].to_numpy()))
            fold_predicted.append(np.maximum(0, np.expm1(validation_log)))
        fold_metrics = [regression_metrics(actual, predicted, target)
                        for actual, predicted in zip(fold_actual, fold_predicted)]
        comparisons.append({
            "model": candidate_name,
            "selection_metric": "r2",
            "cv_metrics": aggregate_fold_metrics(fold_metrics, ["r2", "mae", "rmse"]),
        })

    best = max(comparisons, key=lambda result: result["cv_metrics"]["r2"]["mean"])
    selected_name = best["model"]
    selected_estimator, selected_scale = next(
        (estimator, scale) for name, estimator, scale in regression_candidates()
        if name == selected_name)
    selected = make_pipeline(X_development, clone(selected_estimator), selected_scale)
    selected.fit(X_development, y_development)
    explanation = model_explanation(selected, list(X.columns))
    explanation["permutation"] = cross_validated_permutation_importance(
        X_development, y_development, cv_splits, selected_estimator,
        selected_scale, False)
    test_log = selected.predict(X_test)
    test_predicted = np.maximum(0, np.expm1(test_log))
    test_metrics = regression_metrics(y_test_original, test_predicted, target)

    fold_diagnostics = []
    for fold_number, (train_idx, validation_idx) in enumerate(cv_splits, start=1):
        fold_diagnostics.append({
            "fold": fold_number,
            "training_rows": int(len(train_idx)),
            "validation_rows": int(len(validation_idx)),
            "validation_target_median": float(np.expm1(y_development.iloc[validation_idx]).median()),
        })
    return selected, {
        "model": selected_name,
        "selection_metric": "r2",
        "comparisons": comparisons,
        "metrics": test_metrics,
        "diagnostics": regression_diagnostics(y_test_original, test_predicted),
        "training_rows": int(len(X_development)),
        "test_groups": int(groups.iloc[test_idx].nunique()),
        "features": list(X.columns),
        "explanation": explanation,
        "split_method": "GroupKFold by season and episode",
        "cv_fold_sizes": fold_diagnostics,
    }

def train_regressor(df, X_all, target, model_name, task_name, raw, subset=None):
    data = df if subset is None else df.loc[subset]
    target_values = pd.to_numeric(data[target], errors="coerce")
    mask = target_values.notna() & (target_values >= 0)
    X = X_all.loc[data.index[mask]]
    y_original = target_values.loc[mask]
    y_log = np.log1p(y_original)
    if len(X) < 20: raise ValueError(f"{task_name}: not enough valid training rows ({len(X)})")
    groups = task_groups(raw, X.index)
    model, result = select_regressor(X, y_log, y_original, groups, target)
    joblib.dump(model, MODEL_DIR/f"{model_name}.pkl")
    result.update({"target":target,"target_type":"log_regression"})
    return result

def main():
    if not DATA_PATH.exists(): raise FileNotFoundError("Put data.csv or data(2).csv beside this script.")
    raw = pd.read_csv(DATA_PATH)
    required_cols = ["Accepted Offer","Received Offer","Deal Valuation","Total Deal Amount","Total Deal Equity"]
    missing = [c for c in required_cols if c not in raw.columns]
    if missing: raise ValueError("Dataset is missing required columns: "+", ".join(missing))
    df = clean_frame(raw)
    X = df[features_for(df)].copy()
    results = {}
    print(f"Loaded {len(df)} historical pitches from {DATA_PATH.name}")
    # 1. Deal: missing Accepted Offer means no completed deal, including pitches with no offer.
    df["_deal_target"] = pd.to_numeric(raw["Accepted Offer"],errors="coerce").fillna(0).astype(int)
    results["deal"] = train_classifier(df,X,"_deal_target","deal_model","Deal prediction",raw)
    # 2. Final valuation, 3. capital amount, 4. equity: outcomes only exist for accepted deals.
    accepted = pd.to_numeric(raw["Accepted Offer"],errors="coerce").eq(1)
    results["valuation"] = train_regressor(df.loc[accepted],X.loc[accepted],"Deal Valuation","valuation_model","Deal valuation",raw)
    results["deal_amount"] = train_regressor(df.loc[accepted],X.loc[accepted],"Total Deal Amount","deal_amount_model","Deal amount",raw)
    results["deal_equity"] = train_regressor(df.loc[accepted],X.loc[accepted],"Total Deal Equity","equity_model","Deal equity",raw)
    # 5. Which sharks invest? Train five independent probabilities, conditional on accepted deals.
    shark_metrics = {}
    for shark in SHARKS:
        target_col = f"{shark} Investment Amount"
        if target_col not in raw: continue
        temp = df.loc[accepted].copy()
        temp["_shark_target"] = (pd.to_numeric(raw.loc[accepted,target_col],errors="coerce").fillna(0)>0).astype(int)
        # use accepted deals, but guard against a one-class shark target
        y = temp["_shark_target"]
        if y.nunique() < 2 or y.value_counts().min() < 2:
            print(f"Skipping {shark}: insufficient positive/negative examples")
            continue
        Xs = X.loc[accepted]
        groups = task_groups(raw, Xs.index)
        try:
            selected, shark_result = select_classifier(Xs, y, groups)
        except ValueError as exc:
            print(f"Skipping {shark}: grouped cross-validation unavailable ({exc})")
            continue
        shark_metrics[shark] = shark_result
        joblib.dump(selected,MODEL_DIR/f"shark_{shark.lower()}.pkl")
    results["shark_investment"]={"target":"Individual shark investment","target_type":"multi_label_classification",
        "model":"Validation-selected classification pipelines","metrics":shark_metrics,"training_rows":int(accepted.sum()),"features":list(X.columns)}
    # 6. Founder acceptance is conditional on receiving an offer; this avoids treating no-offer pitches as rejections.
    received = pd.to_numeric(raw["Received Offer"],errors="coerce").eq(1)
    accept_data = raw.loc[received,"Accepted Offer"]
    accept_y = pd.to_numeric(accept_data,errors="coerce")
    valid_accept = accept_y.notna()
    accept_idx = accept_y.index[valid_accept]
    df_accept = df.loc[accept_idx].copy(); df_accept["_accept_target"] = accept_y.loc[accept_idx].astype(int)
    results["founder_acceptance"] = train_classifier(df_accept,X.loc[accept_idx],"_accept_target",
        "acceptance_model","Founder acceptance",raw)
    metadata={"version":"3.0","features":list(X.columns),"sharks":SHARKS,
        "results":results,"currency":"INR lakh","dataset_rows":int(len(df)),
        "accepted_deals":int(accepted.sum()),"received_offers":int(received.sum()),
        "acceptance_scope":"Conditional probability among pitches that received an offer",
        "evaluation_method":"Episode-grouped 80/20 development/test split with five-fold grouped CV on development data. Test data is used only for final evaluation.",
        "feature_availability":{
            "deal":"Pitch-time business, founder, episode, ask/equity and shark-presence fields; post-deal outcomes excluded.",
            "valuation":"Same pitch-time fields; trained only on pitches with a recorded final valuation.",
            "shark_investment":"Same pitch-time fields; shark investment amounts/equity/debt and other deal outcomes excluded.",
            "deal_amount":"Same pitch-time fields; final deal amount and other deal outcomes excluded.",
            "deal_equity":"Same pitch-time fields; final deal equity and other deal outcomes excluded.",
            "founder_acceptance":"Same pitch-time fields among pitches that received an offer; acceptance and deal outcomes excluded from inputs.",
        },
        "excluded_features":{
            "Season Number, Episode Number, Pitch Number":"Excluded from model inputs because episode/pitch identifiers are not captured as actual pitch attributes by the GUI; season/episode are retained only as split groups.",
            "Male Presenters, Female Presenters, Transgender Presenters, Couple Presenters":"Excluded because the current pitch parser/UI does not collect these fields and the previous inference row assumed gender composition.",
            "days_into_season":"Excluded because inference used a fixed placeholder value rather than a user-provided pitch-time value.",
            "Part of Match off":"Excluded because this can reflect post-pitch negotiation and is not reliably available at initial prediction time.",
        },
        "note":"Cross-validation scores guide model selection. Test metrics are from a separate untouched episode-grouped test set and are not guarantees.",
        "source_dataset":DATA_PATH.name}
    joblib.dump(metadata,MODEL_DIR/"metadata.pkl")
    (ROOT/"model_metrics.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print("\nTRAINING COMPLETE — SIX PRESENTATION TASKS")
    for key,item in results.items(): print(f"{key}: {item['metrics']}")
    print(f"Models saved to: {MODEL_DIR}")

if __name__=="__main__": main()
