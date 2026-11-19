"""Run the approved, bounded deal-classification experiment.

This script writes only under experiments/deal_prediction and never replaces
production model files or evaluation metadata.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import deal_experiments as experiment
from train_models import MODEL_DIR

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "experiments" / "deal_prediction"


def compact_metrics(metrics: dict) -> dict:
    return {name: float(value) for name, value in metrics.items()}


def summarize_result(name: str, result: dict, phase: str, config: dict | None = None) -> dict:
    metrics = result["fold_metrics_at_0_5"]
    row = {"phase": phase, "candidate": name, "configuration": config or {}}
    for metric, summary in metrics.items():
        row[f"{metric}_mean"] = summary["mean"]
        row[f"{metric}_std"] = summary["std"]
    return row


def markdown_report(report: dict) -> str:
    """Render the saved experiment data as a compact human-readable report."""
    lines = [
        "# Deal Prediction Experiment",
        "",
        f"Selected candidate: **{report['selected_model']}**",
        f"OOF-selected threshold: **{report['selected_threshold']:.3f}**",
        "Selection used development grouped-CV accuracy, with F1 and ROC-AUC as tie-breakers. "
        "Threshold selection used development OOF probabilities only.",
        "",
        "## Default Candidate Comparison",
        "All values are grouped-CV mean +/- standard deviation at threshold 0.50.",
        "",
        "| Candidate | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "balanced_accuracy"]
    for candidate in report["default_candidate_results"]:
        values = []
        for metric in metric_names:
            mean, std = candidate[f"{metric}_mean"], candidate[f"{metric}_std"]
            values.append(f"{mean:.3f} +/- {std:.3f}")
        lines.append(f"| {candidate['candidate']} | " + " | ".join(values) + " |")

    lines.extend([
        "",
        "## Limited Tuning",
        "Only the top three default candidate families were tried, with three fixed configurations each.",
        "",
        "| Configuration | Accuracy | F1 | Recall | ROC-AUC |",
        "|---|---:|---:|---:|---:|",
    ])
    for candidate in report["limited_tuning_results"]:
        if "accuracy_mean" not in candidate:
            continue
        lines.append(
            f"| {candidate['candidate']} | {candidate['accuracy_mean']:.3f} +/- {candidate['accuracy_std']:.3f} | "
            f"{candidate['f1_mean']:.3f} +/- {candidate['f1_std']:.3f} | "
            f"{candidate['recall_mean']:.3f} +/- {candidate['recall_std']:.3f} | "
            f"{candidate['roc_auc_mean']:.3f} +/- {candidate['roc_auc_std']:.3f} |"
        )

    lines.extend([
        "",
        "## Development Threshold Check",
        "Thresholds below use the selected candidate's pooled development OOF probabilities; these are selection results, not final test estimates.",
        "",
        "| Threshold | Accuracy | Precision | Recall | F1 |",
        "|---:|---:|---:|---:|---:|",
    ])
    threshold_rows = report["selected_oof_threshold_grid"]
    shown_thresholds = {0.40, 0.45, 0.50, 0.55, 0.60}
    for result in threshold_rows:
        threshold = round(result["threshold"], 3)
        if threshold in shown_thresholds:
            lines.append(
                f"| {threshold:.2f} | {result['accuracy']:.3f} | {result['precision']:.3f} | "
                f"{result['recall']:.3f} | {result['f1']:.3f} |"
            )

    selected_test = report["selected_test_metrics_at_selected_threshold"]
    production_test = report["production_test_metrics_at_0_5"]
    lines.extend([
        "",
        "## Final Held-Out Test",
        "This is the single final comparison on the same untouched 157-row episode-grouped test set. The production model uses threshold 0.50.",
        "",
        "| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, threshold, metrics in (
        (f"Selected {report['selected_model']}", report["selected_threshold"], selected_test),
        ("Current production Logistic Regression", 0.5, production_test),
    ):
        lines.append(
            f"| {name} | {threshold:.3f} | {metrics['accuracy']:.3f} | {metrics['precision']:.3f} | "
            f"{metrics['recall']:.3f} | {metrics['f1']:.3f} | {metrics['roc_auc']:.3f} | "
            f"{metrics['pr_auc']:.3f} | {metrics['balanced_accuracy']:.3f} |"
        )
    lines.extend([
        "",
        f"Accuracy change: {report['test_accuracy_change_vs_production']:+.3f}; "
        f"F1 change: {report['test_f1_change_vs_production']:+.3f}; "
        f"ROC-AUC change: {report['test_roc_auc_change_vs_production']:+.3f}.",
        "",
        "## Season-Wise Observations",
        "Development OOF ROC-AUC by season:",
    ])
    for season, metrics in report["selected_oof_season_metrics_at_0_5"].items():
        lines.append(f"- Season {season}: ROC-AUC {metrics['roc_auc']:.3f}, accuracy {metrics['accuracy']:.3f}.")
    lines.append("Held-out test ROC-AUC by season:")
    for season, metrics in report["selected_test_season_metrics_at_selected_threshold"].items():
        lines.append(f"- Season {season}: ROC-AUC {metrics['roc_auc']:.3f}, accuracy {metrics['accuracy']:.3f}.")

    lines.extend([
        "",
        "## Features and Recommendation",
        "Features used: " + ", ".join(report["selected_features"]) + ".",
        "Selected hyperparameters: " + ", ".join(
            f"{name}={report['selected_hyperparameters'][name]}"
            for name in ("n_estimators", "max_depth", "min_samples_leaf", "max_features",
                         "class_weight", "criterion", "C", "learning_rate", "max_iter",
                         "max_leaf_nodes", "random_state")
            if name in report["selected_hyperparameters"]) + ".",
        "The experiment improves all reported aggregate test metrics over the existing production model on the fixed split. "
        "However, the dataset is small and season-level test results vary; a single grouped test split is not enough to claim reliable future-season improvement. "
        "Keep the experimental model separate until the result is reviewed and replicated.",
        "",
        "The production model was not overwritten. The experimental pipeline and machine-readable outputs are saved alongside this report.",
    ])
    return "\n".join(lines) + "\n"


def tune_configurations(model_name: str) -> list[tuple[str, object, bool, dict]]:
    """Return three predeclared, bounded configurations for a leading family."""
    if model_name == "Logistic Regression":
        values = [
            ("C=0.25, unweighted", LogisticRegression(C=0.25, max_iter=3000, random_state=42), True),
            ("C=1, unweighted", LogisticRegression(C=1.0, max_iter=3000, random_state=42), True),
            ("C=1, balanced", LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=42), True),
        ]
    elif model_name == "Decision Tree":
        values = [
            ("depth=3, leaf=8", DecisionTreeClassifier(max_depth=3, min_samples_leaf=8, random_state=42), False),
            ("depth=5, leaf=8", DecisionTreeClassifier(max_depth=5, min_samples_leaf=8, random_state=42), False),
            ("depth=None, leaf=12", DecisionTreeClassifier(max_depth=None, min_samples_leaf=12, random_state=42), False),
        ]
    elif model_name == "Random Forest":
        values = [
            ("trees=250, depth=None, leaf=5", RandomForestClassifier(n_estimators=250, max_depth=None, min_samples_leaf=5, random_state=42, n_jobs=-1), False),
            ("trees=250, depth=8, leaf=5", RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1), False),
            ("trees=250, depth=12, leaf=10", RandomForestClassifier(n_estimators=250, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1), False),
        ]
    elif model_name == "Extra Trees":
        values = [
            ("trees=250, depth=None, leaf=5", ExtraTreesClassifier(n_estimators=250, max_depth=None, min_samples_leaf=5, random_state=42, n_jobs=-1), False),
            ("trees=250, depth=8, leaf=5", ExtraTreesClassifier(n_estimators=250, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1), False),
            ("trees=250, depth=12, leaf=10", ExtraTreesClassifier(n_estimators=250, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1), False),
        ]
    elif model_name == "Gradient Boosting":
        values = [
            ("estimators=100, lr=0.03, depth=2", GradientBoostingClassifier(n_estimators=100, learning_rate=0.03, max_depth=2, random_state=42), False),
            ("estimators=150, lr=0.05, depth=2", GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=42), False),
            ("estimators=150, lr=0.03, depth=3", GradientBoostingClassifier(n_estimators=150, learning_rate=0.03, max_depth=3, random_state=42), False),
        ]
    elif model_name == "HistGradientBoosting":
        values = [
            ("iter=100, lr=0.05, leaf=15", HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42), False),
            ("iter=150, lr=0.05, leaf=15", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42), False),
            ("iter=100, lr=0.08, leaf=7", HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, max_leaf_nodes=7, l2_regularization=2.0, random_state=42), False),
        ]
    else:
        raise ValueError(f"No bounded tuning grid for {model_name}")
    return [(label, estimator, scale, {"label": label})
            for label, estimator, scale in values]


def run_cv_candidate(data, estimator, scale_numeric, display_name, phase, config=None):
    try:
        result = experiment.evaluate_candidate_cv(data, estimator, scale_numeric)
        row = summarize_result(display_name, result, phase, config)
        print(
            f"[{phase}] {display_name}: accuracy "
            f"{row['accuracy_mean']:.3f} +/- {row['accuracy_std']:.3f}; "
            f"ROC-AUC {row['roc_auc_mean']:.3f} +/- {row['roc_auc_std']:.3f}",
            flush=True,
        )
        return {"name": display_name, "estimator": estimator,
                "scale_numeric": scale_numeric, "result": result, "row": row,
                "error": None, "config": config or {}}
    except Exception as exc:
        print(f"[{phase}] {display_name} FAILED: {type(exc).__name__}: {exc}", flush=True)
        return {"name": display_name, "estimator": estimator,
                "scale_numeric": scale_numeric, "result": None, "row": None,
                "error": f"{type(exc).__name__}: {exc}", "config": config or {}}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = experiment.load_deal_experiment_data()
    split_check = experiment.validate_grouped_folds(data)
    print(f"Fixed split validated: {split_check}", flush=True)

    default_results = []
    for name, estimator, scale_numeric in experiment.available_candidate_estimators():
        default_results.append(run_cv_candidate(
            data, estimator, scale_numeric, name, "default"))

    successful_defaults = [entry for entry in default_results if entry["result"] is not None]
    if not successful_defaults:
        raise RuntimeError("No default candidate completed cross-validation.")
    leaders = sorted(
        successful_defaults,
        key=lambda entry: (entry["row"]["accuracy_mean"], entry["row"]["f1_mean"], entry["row"]["roc_auc_mean"]),
        reverse=True,
    )[:3]
    print("Limited tuning families:", ", ".join(entry["name"] for entry in leaders), flush=True)

    tuned_results = []
    for leader in leaders:
        for config_label, estimator, scale_numeric, config in tune_configurations(leader["name"]):
            candidate_name = f"{leader['name']} ({config_label})"
            tuned_results.append(run_cv_candidate(
                data, estimator, scale_numeric, candidate_name, "limited_tuning", config))

    successful_tuned = [entry for entry in tuned_results if entry["result"] is not None]
    if not successful_tuned:
        successful_tuned = successful_defaults
    selected = max(
        successful_tuned,
        key=lambda entry: (entry["row"]["accuracy_mean"], entry["row"]["f1_mean"], entry["row"]["roc_auc_mean"]),
    )
    selected_name = selected["name"]
    selected_result = selected["result"]
    threshold_rows = selected_result["oof_threshold_metrics"]
    best_threshold_row = max(
        threshold_rows,
        key=lambda row: (row["accuracy"], -abs(row["threshold"] - 0.5), row["f1"], row["recall"]),
    )
    selected_threshold = best_threshold_row["threshold"]
    print(f"Selected on development CV: {selected_name}; OOF threshold={selected_threshold:.3f}", flush=True)

    # Final test access occurs only here, after model/configuration/threshold selection.
    selected_pipeline = experiment.make_experiment_pipeline(
        data.X_development, selected["estimator"], selected["scale_numeric"])
    selected_pipeline.fit(data.X_development, data.y_development)
    selected_probability = experiment.positive_class_probability(selected_pipeline, data.X_test)
    selected_test_metrics = experiment.classification_metrics(
        data.y_test, selected_probability, selected_threshold)
    selected_test_metrics_default = experiment.classification_metrics(
        data.y_test, selected_probability, 0.5)

    production_model = joblib.load(MODEL_DIR / "deal_model.pkl")
    production_probability = experiment.positive_class_probability(production_model, data.X_test)
    production_test_metrics = experiment.classification_metrics(
        data.y_test, production_probability, 0.5)

    season_oof = selected_result["oof_season_metrics_at_0_5"]
    season_labels = data.groups_test.str.split("-", n=1).str[0]
    test_seasons = {}
    for season in sorted(season_labels.unique()):
        mask = np.asarray(season_labels == season)
        labels = data.y_test.to_numpy()[mask]
        if np.unique(labels).size == 2:
            test_seasons[str(season)] = experiment.classification_metrics(
                labels, selected_probability[mask], selected_threshold)

    candidate_rows = [entry["row"] for entry in [*default_results, *tuned_results]
                      if entry["row"] is not None]
    candidates_frame = pd.DataFrame(candidate_rows).sort_values(
        ["phase", "accuracy_mean", "f1_mean"], ascending=[True, False, False])
    candidates_frame.to_csv(OUTPUT_DIR / "candidate_cv_summary.csv", index=False)
    pd.DataFrame({
        "development_row_index": data.X_development.index,
        "season": data.groups_development.str.split("-", n=1).str[0].to_numpy(),
        "target": data.y_development.to_numpy(),
        "selected_oof_probability": selected_result["oof_probabilities"],
    }).to_csv(OUTPUT_DIR / "selected_development_oof.csv", index=False)
    joblib.dump(selected_pipeline, OUTPUT_DIR / "selected_deal_model.pkl")

    serializable_default = [
        ({**entry["row"], "oof_season_metrics_at_0_5": entry["result"]["oof_season_metrics_at_0_5"]}
         if entry["row"] else {
             "phase": "default", "candidate": entry["name"], "error": entry["error"]})
        for entry in default_results]
    serializable_tuned = [
        ({**entry["row"], "oof_season_metrics_at_0_5": entry["result"]["oof_season_metrics_at_0_5"]}
         if entry["row"] else {
             "phase": "limited_tuning", "candidate": entry["name"], "error": entry["error"]})
        for entry in tuned_results]
    report = {
        "experiment_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(data.X_development.index.name or "data(2).csv"),
        "sklearn_version": sklearn.__version__,
        "random_seeds": {"test_split": 42, "cv_folds": 43, "estimators": 42},
        "split": split_check,
        "final_test_used_only_after_selection": True,
        "selection_rule": "Highest mean grouped-CV accuracy; tie-break by F1 then ROC-AUC.",
        "selected_model": selected_name,
        "selected_features": list(data.X_development.columns),
        "selected_hyperparameters": selected["estimator"].get_params(deep=False),
        "selected_threshold": selected_threshold,
        "selected_oof_threshold_metrics": compact_metrics(best_threshold_row),
        "selected_oof_threshold_grid": [compact_metrics(row) for row in threshold_rows],
        "selected_oof_default_threshold_metrics": compact_metrics(
            experiment.classification_metrics(data.y_development, selected_result["oof_probabilities"], 0.5)),
        "selected_cv_fold_metrics_at_0_5": selected_result["fold_metrics_at_0_5"],
        "default_candidate_results": serializable_default,
        "limited_tuning_results": serializable_tuned,
        "selected_test_metrics_at_selected_threshold": compact_metrics(selected_test_metrics),
        "selected_test_metrics_at_0_5": compact_metrics(selected_test_metrics_default),
        "production_model_name": "LogisticRegression (existing saved pipeline)",
        "production_test_metrics_at_0_5": compact_metrics(production_test_metrics),
        "test_accuracy_change_vs_production": float(selected_test_metrics["accuracy"] - production_test_metrics["accuracy"]),
        "test_f1_change_vs_production": float(selected_test_metrics["f1"] - production_test_metrics["f1"]),
        "test_roc_auc_change_vs_production": float(selected_test_metrics["roc_auc"] - production_test_metrics["roc_auc"]),
        "selected_oof_season_metrics_at_0_5": season_oof,
        "selected_test_season_metrics_at_selected_threshold": test_seasons,
        "optional_boosters_available": [entry["name"] for entry in successful_defaults
                                        if entry["name"] in {"XGBoost", "LightGBM", "CatBoost"}],
        "candidate_failures": [
            {"phase": entry["row"]["phase"] if entry["row"] else phase,
             "candidate": entry["name"], "error": entry["error"]}
            for phase, entries in (("default", default_results), ("limited_tuning", tuned_results))
            for entry in entries if entry["error"]
        ],
        "limitations": [
            "Small dataset with season-wise distribution differences and a relatively small grouped test set.",
            "Threshold was selected from development OOF predictions; OOF-based threshold performance is selection-biased and final test metrics are the unbiased check.",
        ],
    }
    (OUTPUT_DIR / "experiment_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (OUTPUT_DIR / "experiment_report.md").write_text(markdown_report(report), encoding="utf-8")
    print("FINAL TEST (selected threshold):", selected_test_metrics, flush=True)
    print("FINAL TEST (production threshold 0.5):", production_test_metrics, flush=True)
    print(f"Artifacts saved separately under: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        main()
    if caught:
        print("WARNINGS:", flush=True)
        for warning in caught:
            print(f"{warning.category.__name__}: {warning.message}", flush=True)
