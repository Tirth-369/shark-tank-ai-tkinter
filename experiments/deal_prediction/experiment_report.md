# Deal Prediction Experiment

Selected candidate: **Random Forest (trees=250, depth=8, leaf=5)**
OOF-selected threshold: **0.500**
Selection used development grouped-CV accuracy, with F1 and ROC-AUC as tie-breakers. Threshold selection used development OOF probabilities only.

## Default Candidate Comparison
All values are grouped-CV mean +/- standard deviation at threshold 0.50.

| Candidate | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.589 +/- 0.021 | 0.611 +/- 0.014 | 0.695 +/- 0.064 | 0.650 +/- 0.029 | 0.617 +/- 0.056 | 0.668 +/- 0.064 | 0.576 +/- 0.021 |
| Decision Tree | 0.543 +/- 0.025 | 0.589 +/- 0.025 | 0.563 +/- 0.035 | 0.575 +/- 0.026 | 0.540 +/- 0.026 | 0.572 +/- 0.015 | 0.540 +/- 0.026 |
| Random Forest | 0.590 +/- 0.029 | 0.610 +/- 0.026 | 0.718 +/- 0.049 | 0.658 +/- 0.023 | 0.610 +/- 0.040 | 0.649 +/- 0.025 | 0.575 +/- 0.033 |
| Extra Trees | 0.590 +/- 0.033 | 0.614 +/- 0.025 | 0.690 +/- 0.042 | 0.649 +/- 0.029 | 0.615 +/- 0.048 | 0.649 +/- 0.037 | 0.579 +/- 0.034 |
| Gradient Boosting | 0.592 +/- 0.020 | 0.615 +/- 0.015 | 0.692 +/- 0.040 | 0.651 +/- 0.021 | 0.610 +/- 0.024 | 0.635 +/- 0.027 | 0.580 +/- 0.022 |
| HistGradientBoosting | 0.582 +/- 0.033 | 0.608 +/- 0.027 | 0.684 +/- 0.030 | 0.643 +/- 0.026 | 0.617 +/- 0.028 | 0.654 +/- 0.021 | 0.571 +/- 0.035 |

## Limited Tuning
Only the top three default candidate families were tried, with three fixed configurations each.

| Configuration | Accuracy | F1 | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Gradient Boosting (estimators=100, lr=0.03, depth=2) | 0.558 +/- 0.048 | 0.644 +/- 0.035 | 0.724 +/- 0.044 | 0.609 +/- 0.044 |
| Gradient Boosting (estimators=150, lr=0.05, depth=2) | 0.573 +/- 0.052 | 0.642 +/- 0.045 | 0.698 +/- 0.062 | 0.609 +/- 0.039 |
| Gradient Boosting (estimators=150, lr=0.03, depth=3) | 0.589 +/- 0.041 | 0.662 +/- 0.035 | 0.733 +/- 0.050 | 0.606 +/- 0.028 |
| Random Forest (trees=250, depth=None, leaf=5) | 0.589 +/- 0.032 | 0.662 +/- 0.021 | 0.733 +/- 0.035 | 0.628 +/- 0.041 |
| Random Forest (trees=250, depth=8, leaf=5) | 0.601 +/- 0.041 | 0.675 +/- 0.023 | 0.750 +/- 0.033 | 0.631 +/- 0.047 |
| Random Forest (trees=250, depth=12, leaf=10) | 0.600 +/- 0.027 | 0.675 +/- 0.020 | 0.756 +/- 0.043 | 0.635 +/- 0.048 |
| Extra Trees (trees=250, depth=None, leaf=5) | 0.576 +/- 0.023 | 0.667 +/- 0.025 | 0.773 +/- 0.048 | 0.605 +/- 0.039 |
| Extra Trees (trees=250, depth=8, leaf=5) | 0.570 +/- 0.040 | 0.673 +/- 0.037 | 0.808 +/- 0.065 | 0.597 +/- 0.043 |
| Extra Trees (trees=250, depth=12, leaf=10) | 0.582 +/- 0.028 | 0.683 +/- 0.023 | 0.819 +/- 0.050 | 0.608 +/- 0.035 |

## Development Threshold Check
Thresholds below use the selected candidate's pooled development OOF probabilities; these are selection results, not final test estimates.

| Threshold | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 0.40 | 0.590 | 0.579 | 0.934 | 0.715 |
| 0.45 | 0.600 | 0.594 | 0.862 | 0.703 |
| 0.50 | 0.601 | 0.613 | 0.750 | 0.674 |
| 0.55 | 0.593 | 0.637 | 0.606 | 0.622 |
| 0.60 | 0.566 | 0.662 | 0.434 | 0.524 |

## Final Held-Out Test
This is the single final comparison on the same untouched 157-row episode-grouped test set. The production model uses threshold 0.50.

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Selected Random Forest (trees=250, depth=8, leaf=5) | 0.500 | 0.611 | 0.611 | 0.802 | 0.693 | 0.624 | 0.661 | 0.591 |
| Current production Logistic Regression | 0.500 | 0.497 | 0.538 | 0.570 | 0.554 | 0.448 | 0.507 | 0.489 |

Accuracy change: +0.115; F1 change: +0.140; ROC-AUC change: +0.175.

## Season-Wise Observations
Development OOF ROC-AUC by season:
- Season 1: ROC-AUC 0.637, accuracy 0.631.
- Season 2: ROC-AUC 0.554, accuracy 0.551.
- Season 3: ROC-AUC 0.688, accuracy 0.677.
- Season 4: ROC-AUC 0.661, accuracy 0.611.
- Season 5: ROC-AUC 0.591, accuracy 0.547.
Held-out test ROC-AUC by season:
- Season 1: ROC-AUC 0.564, accuracy 0.490.
- Season 2: ROC-AUC 0.569, accuracy 0.636.
- Season 3: ROC-AUC 0.803, accuracy 0.741.
- Season 4: ROC-AUC 0.431, accuracy 0.533.
- Season 5: ROC-AUC 0.889, accuracy 0.833.

## Features and Recommendation
Features used: Started in, Number of Presenters, Pitchers Average Age, Yearly Revenue, Monthly Sales, Gross Margin, Net Margin, EBITDA, Cash Burn, SKUs, Has Patents, Original Ask Amount, Original Offered Equity, Valuation Requested, Namita Present, Vineeta Present, Anupam Present, Aman Present, Peyush Present, Ritesh Present, Amit Present, Guest Present, ask_log, valuation_requested_log, ask_equity_ratio, valuation_ask_ratio, sharks_present_count, Industry, Pitchers State, Bootstrapped.
Selected hyperparameters: n_estimators=250, max_depth=8, min_samples_leaf=5, max_features=sqrt, class_weight=None, criterion=gini, max_leaf_nodes=None, random_state=42.
The experiment improves all reported aggregate test metrics over the existing production model on the fixed split. However, the dataset is small and season-level test results vary; a single grouped test split is not enough to claim reliable future-season improvement. Keep the experimental model separate until the result is reviewed and replicated.

The production model was not overwritten. The experimental pipeline and machine-readable outputs are saved alongside this report.
