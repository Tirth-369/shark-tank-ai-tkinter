"""Preparation utilities for reproducible deal-classifier experiments.

This module is intentionally separate from production training and prediction.
It never writes model files or uses the final test partition for selection.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import train_models as training


@dataclass(frozen=True)
class DealExperimentData:
    """One fixed episode-grouped split and shared development CV folds."""

    X_development: pd.DataFrame
    y_development: pd.Series
    groups_development: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    groups_test: pd.Series
    cv_splits: tuple[tuple[np.ndarray, np.ndarray], ...]


def load_deal_experiment_data(raw: pd.DataFrame | None = None) -> DealExperimentData:
    """Prepare aligned deal rows without fitting preprocessing or touching test labels."""
    if raw is None:
        raw = pd.read_csv(training.DATA_PATH)
    cleaned = training.clean_frame(raw)
    X = cleaned[training.features_for(cleaned)].copy()
    y = pd.to_numeric(raw["Accepted Offer"], errors="coerce").fillna(0).astype(int)
    groups = training.task_groups(raw, X.index)
    development_idx, test_idx = training.development_test_indices(y, groups, True)

    X_development = X.iloc[development_idx].copy()
    y_development = y.iloc[development_idx].copy()
    groups_development = groups.iloc[development_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_test = y.iloc[test_idx].copy()
    groups_test = groups.iloc[test_idx].copy()
    folds = tuple(
        (np.asarray(train_idx), np.asarray(validation_idx))
        for train_idx, validation_idx in training.development_cv_splits(
            y_development, groups_development, True)
    )
    return DealExperimentData(
        X_development=X_development,
        y_development=y_development,
        groups_development=groups_development,
        X_test=X_test,
        y_test=y_test,
        groups_test=groups_test,
        cv_splits=folds,
    )


def available_candidate_estimators() -> list[tuple[str, Any, bool]]:
    """Return supported installed candidates; bool indicates numeric scaling."""
    candidates: list[tuple[str, Any, bool]] = [
        ("Logistic Regression", LogisticRegression(max_iter=3000, random_state=42), True),
        ("Decision Tree", DecisionTreeClassifier(random_state=42), False),
        ("Random Forest", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1), False),
        ("Extra Trees", ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1), False),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=42), False),
        ("HistGradientBoosting", HistGradientBoostingClassifier(random_state=42), False),
    ]
    optional = {
        "xgboost": ("XGBoost", "XGBClassifier"),
        "lightgbm": ("LightGBM", "LGBMClassifier"),
        "catboost": ("CatBoost", "CatBoostClassifier"),
    }
    for module_name, (display_name, _) in optional.items():
        if importlib.util.find_spec(module_name) is None:
            continue
        if module_name == "xgboost":
            from xgboost import XGBClassifier
            estimator = XGBClassifier(
                n_estimators=250, max_depth=3, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85,
                eval_metric="logloss", random_state=42, n_jobs=-1,
            )
        elif module_name == "lightgbm":
            from lightgbm import LGBMClassifier
            estimator = LGBMClassifier(
                n_estimators=250, max_depth=-1, learning_rate=0.05,
                num_leaves=15, random_state=42, n_jobs=-1, verbosity=-1,
            )
        else:
            from catboost import CatBoostClassifier
            estimator = CatBoostClassifier(
                iterations=250, depth=4, learning_rate=0.05,
                random_seed=42, verbose=False, allow_writing_files=False,
            )
        candidates.append((display_name, estimator, False))
    return candidates


def make_experiment_pipeline(X: pd.DataFrame, estimator: Any,
                             scale_numeric: bool) -> Pipeline:
    """Create fold-fitted preprocessing; dense one-hot output supports HistGB."""
    numeric = [column for column in X.columns if column in training.NUMERIC]
    categorical = [column for column in X.columns if column in training.CATEGORICAL]
    preprocess = ColumnTransformer(
        [
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler() if scale_numeric else "passthrough"),
            ]), numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return Pipeline([("preprocess", preprocess), ("model", clone(estimator))])


def positive_class_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Extract P(target=1) by class label, never by assumed column position."""
    probabilities = model.predict_proba(X)
    classes = list(model.named_steps["model"].classes_)
    if 1 not in classes:
        raise ValueError("Fitted candidate does not contain positive class 1.")
    return probabilities[:, classes.index(1)]


def generate_oof_probabilities(X: pd.DataFrame, y: pd.Series,
                               cv_splits: tuple[tuple[np.ndarray, np.ndarray], ...],
                               estimator: Any, scale_numeric: bool) -> np.ndarray:
    """Generate one out-of-fold probability per development row on shared folds."""
    probabilities = np.full(len(y), np.nan, dtype=float)
    validation_counts = np.zeros(len(y), dtype=int)
    for train_idx, validation_idx in cv_splits:
        fold_model = make_experiment_pipeline(X.iloc[train_idx], estimator, scale_numeric)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probabilities[validation_idx] = positive_class_probability(
            fold_model, X.iloc[validation_idx])
        validation_counts[validation_idx] += 1
    if not np.all(validation_counts == 1):
        raise ValueError("CV folds must validate every development row exactly once.")
    if not np.isfinite(probabilities).all():
        raise ValueError("OOF predictions contain missing or non-finite probabilities.")
    return probabilities


def classification_metrics(y_true: pd.Series | np.ndarray,
                            probabilities: np.ndarray,
                            threshold: float = 0.5) -> dict[str, float]:
    """Compute requested classification metrics without changing labels/scores."""
    y_values = np.asarray(y_true, dtype=int)
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_values, predicted)),
        "precision": float(precision_score(y_values, predicted, zero_division=0)),
        "recall": float(recall_score(y_values, predicted, zero_division=0)),
        "f1": float(f1_score(y_values, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_values, probabilities)),
        "pr_auc": float(average_precision_score(y_values, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(y_values, predicted)),
    }


def threshold_accuracy_summary(y_true: pd.Series | np.ndarray,
                               probabilities: np.ndarray,
                               thresholds: tuple[float, ...] = tuple(np.arange(0.30, 0.701, 0.025))) -> list[dict[str, float]]:
    """Summarize development OOF threshold tradeoffs for later review."""
    return [
        {"threshold": float(threshold), **classification_metrics(y_true, probabilities, float(threshold))}
        for threshold in thresholds
    ]


def _metric_summary(fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = fold_metrics[0].keys()
    return {
        key: {
            "mean": float(np.mean([metrics[key] for metrics in fold_metrics])),
            "std": float(np.std([metrics[key] for metrics in fold_metrics], ddof=1)),
        }
        for key in keys
    }


def evaluate_candidate_cv(data: DealExperimentData, estimator: Any,
                         scale_numeric: bool) -> dict[str, Any]:
    """Evaluate one candidate on the shared folds and return development-only outputs.

    The final test partition is deliberately not referenced by this function.
    """
    oof_probability = np.full(len(data.y_development), np.nan, dtype=float)
    fold_metrics = []
    for train_idx, validation_idx in data.cv_splits:
        fold_model = make_experiment_pipeline(
            data.X_development.iloc[train_idx], estimator, scale_numeric)
        fold_model.fit(data.X_development.iloc[train_idx], data.y_development.iloc[train_idx])
        fold_probability = positive_class_probability(
            fold_model, data.X_development.iloc[validation_idx])
        oof_probability[validation_idx] = fold_probability
        fold_metrics.append(classification_metrics(
            data.y_development.iloc[validation_idx], fold_probability, threshold=0.5))

    if not np.isfinite(oof_probability).all():
        raise ValueError("Candidate did not produce finite OOF probabilities for every development row.")
    season_summary = {}
    row_seasons = data.groups_development.str.split("-", n=1).str[0]
    for season in sorted(row_seasons.unique()):
        row_mask = np.asarray(row_seasons == season)
        if not row_mask.any():
            continue
        season_labels = data.y_development.to_numpy()[row_mask]
        if np.unique(season_labels).size == 2:
            season_summary[str(season)] = classification_metrics(
                season_labels, oof_probability[row_mask], threshold=0.5)

    return {
        "fold_metrics_at_0_5": _metric_summary(fold_metrics),
        "oof_probabilities": oof_probability,
        "oof_threshold_metrics": threshold_accuracy_summary(
            data.y_development.to_numpy(), oof_probability),
        "oof_season_metrics_at_0_5": season_summary,
    }


def validate_grouped_folds(data: DealExperimentData) -> dict[str, Any]:
    """Validate disjoint groups, full OOF coverage, and identical fold indices."""
    test_groups = set(data.groups_test)
    development_groups = set(data.groups_development)
    if test_groups & development_groups:
        raise ValueError("Episode groups overlap between development and final test data.")
    coverage = np.zeros(len(data.y_development), dtype=int)
    for train_idx, validation_idx in data.cv_splits:
        train_groups = set(data.groups_development.iloc[train_idx])
        validation_groups = set(data.groups_development.iloc[validation_idx])
        if train_groups & validation_groups:
            raise ValueError("Episode groups overlap within a CV fold.")
        coverage[validation_idx] += 1
    if not np.all(coverage == 1):
        raise ValueError("CV folds do not provide exactly one validation prediction per row.")
    return {
        "development_rows": int(len(data.y_development)),
        "test_rows": int(len(data.y_test)),
        "development_groups": len(development_groups),
        "test_groups": len(test_groups),
        "fold_count": len(data.cv_splits),
        "test_groups_disjoint": True,
        "oof_coverage_exactly_once": True,
    }
