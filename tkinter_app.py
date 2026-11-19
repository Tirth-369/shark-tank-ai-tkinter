"""Simple Shark Tank AI GUI.

Tkinter widgets and layouts are limited to the concepts covered in L10:
Tk, Frame, Label, Entry, Text, Button and Checkbutton, using pack/grid.
The application structure applies the SOLID design principles from L11.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

import joblib
import numpy as np
import pandas as pd

from pitch_extractor import extract_pitch

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
SHARKS = ["Namita", "Vineeta", "Anupam", "Aman", "Peyush"]
PANEL_SHARKS = [*SHARKS, "Ritesh", "Amit", "Guest"]

# -------------------- SOLID: Interface Segregation --------------------
class ModelReader(ABC):
    """Small interface: read a named trained model."""
    @abstractmethod
    def get_model(self, name: str):
        raise NotImplementedError


class MetadataReader(ABC):
    """Small interface: read model metadata separately."""
    @abstractmethod
    def get_metadata(self) -> dict:
        raise NotImplementedError


# -------------------- SOLID: Single Responsibility --------------------
class LocalModelStore(ModelReader, MetadataReader):
    """Only responsible for loading trained model files from disk."""
    def __init__(self, folder: Path):
        self.folder = folder
        self._models = {}
        self._metadata = None

    def get_model(self, name: str):
        if name not in self._models:
            path = self.folder / f"{name}.pkl"
            if not path.is_file():
                raise FileNotFoundError(f"Missing model: {path.name}. Run run_train_tkinter.bat first.")
            self._models[name] = joblib.load(path)
        return self._models[name]

    def get_metadata(self) -> dict:
        if self._metadata is None:
            path = self.folder / "metadata.pkl"
            if not path.is_file():
                raise FileNotFoundError("Missing models/metadata.pkl. Run run_train_tkinter.bat first.")
            self._metadata = joblib.load(path)
        return self._metadata


class PitchTextParser:
    """Only responsible for extracting structured details from pitch text."""
    def parse(self, text: str) -> dict:
        return extract_pitch(text)


class PitchPredictionService:
    """Only responsible for preparing model input and generating predictions.

    DIP: this service receives small reader abstractions, not a concrete UI or
    file-loading implementation. Composition is performed in main().
    """
    def __init__(self, model_reader: ModelReader, metadata_reader: MetadataReader):
        self.model_reader = model_reader
        self.metadata_reader = metadata_reader
        self.defaults = self._make_defaults()

    def _make_defaults(self):
        return {
            "industry": "Unknown", "state": "Unknown",
            "ask": np.nan, "equity": np.nan, "valuation": np.nan,
            "revenue": np.nan, "monthly_sales": np.nan,
            "gross_margin": np.nan, "net_margin": np.nan, "ebitda": np.nan,
            "started_in": np.nan, "presenters": np.nan, "age": np.nan,
            "skus": np.nan,
            "bootstrapped": "unknown", "patents": 0, "cash_burn": 0,
            "panel_presence": {shark: None for shark in PANEL_SHARKS},
        }

    def prepare_pitch(self, extracted: dict) -> dict:
        """Use extracted values and preserve missing numeric values for pipeline imputation."""
        p = dict(self.defaults)
        mapping = {
            "industry": "industry", "state": "state", "ask": "ask", "equity": "equity",
            "valuation": "valuation", "revenue": "revenue", "monthly_sales": "monthly_sales",
            "gross_margin": "gross_margin", "net_margin": "net_margin", "ebitda": "ebitda",
            "started_in": "started_in", "presenters": "presenters", "age": "age", "skus": "skus",
            "bootstrapped": "bootstrapped", "patents": "patents", "cash_burn": "cash_burn"
        }
        estimated = []
        for source, target in mapping.items():
            value = extracted.get(source)
            if value is None:
                estimated.append(target)
            else:
                p[target] = value
        if "bootstrapped" not in extracted or extracted.get("bootstrapped") is None:
            p["bootstrapped"] = "unknown"
        else:
            p["bootstrapped"] = "yes" if bool(extracted["bootstrapped"]) else "funded"
        if extracted.get("valuation") is None and extracted.get("ask") is not None and (extracted.get("equity") or 0) > 0:
            p["valuation"] = extracted["ask"] * 100 / extracted["equity"]
            estimated.append("valuation (derived from ask/equity)")
        if extracted.get("monthly_sales") is None and extracted.get("revenue") is not None:
            p["monthly_sales"] = extracted["revenue"] / 12
            estimated.append("monthly sales (derived from annual revenue)")
        p["estimated_fields"] = estimated
        return p

    @staticmethod
    def _positive_probability(model, row):
        probabilities = model.predict_proba(row)
        classes = list(getattr(model, "classes_", []))
        if not classes and hasattr(model, "named_steps"):
            classes = list(getattr(model.named_steps.get("model"), "classes_", []))
        return float(probabilities[0, classes.index(1)]) if 1 in classes else 0.0

    def _build_model_row(self, p: dict) -> pd.DataFrame:
        panel_presence = p.get("panel_presence", {})
        presence_features = {
            f"{shark} Present": (np.nan if panel_presence.get(shark) is None
                                 else int(panel_presence[shark]))
            for shark in PANEL_SHARKS
        }
        known_presence = [value for value in panel_presence.values() if value is not None]
        panel_count = sum(bool(value) for value in known_presence) if len(known_presence) == 8 else np.nan
        row = {
            "Season Number": 5, "Episode Number": 10, "Pitch Number": 3,
            "Started in": p["started_in"], "Number of Presenters": p["presenters"],
            "Male Presenters": p["presenters"], "Female Presenters": 0,
            "Transgender Presenters": 0, "Couple Presenters": 0,
            "Pitchers Average Age": p["age"], "Yearly Revenue": p["revenue"],
            "Monthly Sales": p["monthly_sales"], "Gross Margin": p["gross_margin"],
            "Net Margin": p["net_margin"], "EBITDA": p["ebitda"],
            "Cash Burn": p["cash_burn"], "SKUs": p["skus"], "Has Patents": p["patents"],
            "Part of Match off": 0, "Original Ask Amount": p["ask"],
            "Original Offered Equity": p["equity"], "Valuation Requested": p["valuation"],
            **presence_features,
            "Industry": p["industry"], "Pitchers State": p["state"],
            "Bootstrapped": p.get("bootstrapped", "unknown"),
            "days_into_season": 30,
            "ask_log": np.log1p(max(0, p["ask"])) if np.isfinite(p["ask"]) else np.nan,
            "valuation_requested_log": np.log1p(max(0, p["valuation"])) if np.isfinite(p["valuation"]) else np.nan,
            "ask_equity_ratio": p["equity"] / (p["ask"] + 1),
            "valuation_ask_ratio": p["valuation"] / (p["ask"] + 1),
            "sharks_present_count": panel_count,
        }
        return pd.DataFrame([row])

    def predict(self, p: dict, selected_sharks: list[str]) -> dict:
        meta = self.metadata_reader.get_metadata()
        row = self._build_model_row(p)
        features = meta.get("features", [])
        if features:
            missing = [column for column in features if column not in row.columns]
            if missing:
                raise ValueError("Missing model input features: " + ", ".join(missing))
            row = row.loc[:, features]

        # The result keys map directly to the six tasks in the project presentation.
        return {
            "deal": self._positive_probability(self.model_reader.get_model("deal_model"), row),
            "valuation": max(0.0, float(np.expm1(self.model_reader.get_model("valuation_model").predict(row)[0]))),
            "sharks": {
                shark: self._positive_probability(self.model_reader.get_model(f"shark_{shark.lower()}"), row)
                for shark in selected_sharks
            },
            "amount": max(0.0, float(np.expm1(self.model_reader.get_model("deal_amount_model").predict(row)[0]))),
            "equity": float(np.clip(np.expm1(self.model_reader.get_model("equity_model").predict(row)[0]), 0, 100)),
            "acceptance": self._positive_probability(self.model_reader.get_model("acceptance_model"), row),
        }


# -------------------- SOLID: Presentation only --------------------
class SharkTankGUI:
    """Only responsible for Tkinter widgets, layout and user events."""
    BG = "#f4f6f8"
    PANEL = "#ffffff"
    TEXT = "#20252b"
    MUTED = "#5e6872"
    BLUE = "#245b8f"

    # OCP: output rows are data-driven; another result can be added here without
    # rewriting the widget-building logic.
    RESULT_ROWS = [
        ("Deal prediction", "deal"),
        ("Final deal valuation", "valuation"),
        ("Individual shark investment", "sharks"),
        ("Total deal amount", "amount"),
        ("Total deal equity", "equity"),
        ("Founder acceptance", "acceptance"),
    ]

    def __init__(self, root: tk.Tk, parser: PitchTextParser,
                 prediction_service: PitchPredictionService):
        self.root = root
        self.parser = parser
        self.prediction_service = prediction_service
        self.extracted = {}
        self.pitch_values = {}
        self.flag_vars = {}
        self.shark_vars = {}
        self._model_analysis_window = None
        self._build_window()

    def _build_window(self):
        self.root.title("Shark Tank AI - Pitch Predictor")
        self.root.geometry("1050x850")
        self.root.minsize(900, 760)
        self.root.configure(bg=self.BG)

        heading = tk.Frame(self.root, bg=self.BG)
        heading.pack(fill="x", padx=22, pady=(16, 8))
        tk.Label(heading, text="Shark Tank AI", bg=self.BG, fg=self.TEXT,
                 font=("Arial", 22, "bold")).pack(anchor="w")
        tk.Label(heading, text="Enter a startup pitch and view the six project predictions.",
                 bg=self.BG, fg=self.MUTED, font=("Arial", 10)).pack(anchor="w", pady=(3, 0))

        pitch_frame = tk.Frame(self.root, bg=self.PANEL, bd=1, relief="solid")
        pitch_frame.pack(fill="x", padx=22, pady=8)
        tk.Label(pitch_frame, text="Startup pitch", bg=self.PANEL, fg=self.TEXT,
                 font=("Arial", 11, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        self.pitch_box = tk.Text(pitch_frame, height=5, wrap="word", font=("Arial", 10),
                                 relief="solid", bd=1)
        self.pitch_box.pack(fill="x", padx=12, pady=(0, 8))
        actions = tk.Frame(pitch_frame, bg=self.PANEL)
        actions.pack(anchor="w", padx=12, pady=(0, 10))
        self._button(actions, "Load sample", self.load_sample).pack(side="left", padx=(0, 6))
        self._button(actions, "Extract details", self.extract_details).pack(side="left", padx=6)
        self._button(actions, "Analyze pitch", self.analyze_pitch, primary=True).pack(side="left", padx=6)
        self._button(actions, "Model details", self.show_model_details).pack(side="left", padx=6)
        self._button(actions, "Clear", self.clear_form).pack(side="left", padx=6)

        details = tk.Frame(self.root, bg=self.PANEL, bd=1, relief="solid")
        details.pack(fill="x", padx=22, pady=8)
        tk.Label(details, text="Pitch details (check or edit before analysis)", bg=self.PANEL,
                 fg=self.TEXT, font=("Arial", 11, "bold")).grid(row=0, column=0, columnspan=3,
                                                                 sticky="w", padx=12, pady=(10, 6))
        fields = [
            ("Industry", "industry"), ("Founder state", "state"),
            ("Ask amount (₹ lakh)", "ask"), ("Equity offered (%)", "equity"),
            ("Requested valuation (₹ lakh)", "valuation"), ("Annual revenue (₹ lakh)", "revenue"),
            ("Monthly sales (₹ lakh)", "monthly_sales"), ("Gross margin (%)", "gross_margin"),
            ("Net margin (%)", "net_margin"),
        ]
        for index, (label, key) in enumerate(fields):
            row, column = 1 + index // 3, index % 3
            cell = tk.Frame(details, bg=self.PANEL)
            cell.grid(row=row, column=column, sticky="ew", padx=10, pady=5)
            tk.Label(cell, text=label, bg=self.PANEL, fg=self.MUTED,
                     font=("Arial", 9)).pack(anchor="w")
            variable = tk.StringVar()
            self.pitch_values[key] = variable
            tk.Entry(cell, textvariable=variable, width=27, font=("Arial", 10),
                     relief="solid", bd=1).pack(fill="x", ipady=4)
        for column in range(3):
            details.grid_columnconfigure(column, weight=1)

        options = tk.Frame(details, bg=self.PANEL)
        options.grid(row=4, column=0, columnspan=3, sticky="w", padx=10, pady=(8, 4))
        tk.Label(options, text="Business details:", bg=self.PANEL, fg=self.TEXT,
                 font=("Arial", 9, "bold")).pack(side="left", padx=(0, 6))
        for label, key in [("Patent / IP", "patents"), ("Cash burn", "cash_burn")]:
            variable = tk.BooleanVar(value=False)
            self.flag_vars[key] = variable
            tk.Checkbutton(options, text=label, variable=variable, bg=self.PANEL,
                           activebackground=self.PANEL).pack(side="left", padx=5)

        shark_frame = tk.Frame(details, bg=self.PANEL)
        shark_frame.grid(row=5, column=0, columnspan=3, sticky="w", padx=10, pady=(4, 10))
        tk.Label(shark_frame, text="Sharks whose investment predictions to show:",
                 bg=self.PANEL, fg=self.TEXT, font=("Arial", 9, "bold")).pack(side="left", padx=(0, 6))
        for shark in SHARKS:
            variable = tk.BooleanVar(value=True)
            self.shark_vars[shark] = variable
            tk.Checkbutton(shark_frame, text=shark, variable=variable, bg=self.PANEL,
                           activebackground=self.PANEL).pack(side="left", padx=4)

        result_frame = tk.Frame(self.root, bg=self.PANEL, bd=1, relief="solid")
        result_frame.pack(fill="both", expand=True, padx=22, pady=(8, 14))
        tk.Label(result_frame, text="Prediction results", bg=self.PANEL, fg=self.TEXT,
                 font=("Arial", 11, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        self.result_box = tk.Text(result_frame, height=9, wrap="word", font=("Consolas", 10),
                                  relief="solid", bd=1, state="disabled")
        self.result_box.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        self.status = tk.Label(self.root, text="Ready", bg=self.BG, fg=self.MUTED,
                               font=("Arial", 9), anchor="w")
        self.status.pack(fill="x", padx=24, pady=(0, 8))

    def _button(self, parent, text, command, primary=False):
        return tk.Button(parent, text=text, command=command,
                         bg=self.BLUE if primary else "#e7edf3",
                         fg="white" if primary else self.TEXT,
                         activebackground="#17466f", activeforeground="white",
                         relief="raised", padx=12, pady=5, font=("Arial", 9, "bold"))

    def _format_prediction_results(self, result, pitch):
        lines = []
        formatters = {
            "deal": lambda number, title: f"{number}. {title}: {result['deal'] * 100:.1f}% probability",
            "valuation": lambda number, title: f"{number}. {title}: {self._money(result['valuation'])}",
            "amount": lambda number, title: f"{number}. {title}: {self._money(result['amount'])}",
            "equity": lambda number, title: f"{number}. {title}: {result['equity']:.1f}%",
            "acceptance": lambda number, title: f"{number}. {title}: {result['acceptance'] * 100:.1f}% probability",
        }
        for number, (title, key) in enumerate(self.RESULT_ROWS, start=1):
            if key == "sharks":
                lines.append(f"{number}. {title}:")
                lines.extend(
                    f"      {shark:<12} {score * 100:.1f}% probability"
                    for shark, score in result["sharks"].items()
                )
            else:
                lines.append(formatters[key](number, title))
        lines.extend([
            "",
            "Note: Founder acceptance is conditional on receiving an offer.",
            "Shark scores are conditional on an accepted deal. Valuation, amount and equity",
            "are estimates for a completed deal. All results are historical-pattern estimates, not guarantees.",
            "Currency values are shown in INR lakh.",
        ])
        estimated = pitch.get("estimated_fields", [])
        if estimated:
            lines.extend(["", "Fields estimated from dataset statistics or derived: " + ", ".join(estimated)])
        return "\n".join(lines)

    def load_sample(self):
        sample = ("We are building a D2C healthy snack brand in Gujarat for young professionals. "
                  "We have been operating since 2023 and generate annual revenue of ₹50 lakh. "
                  "We are asking for ₹50 lakh in exchange for 5% equity at a requested valuation "
                  "of ₹10 crore. Our products are sold through our website and retail stores. "
                  "We have strong gross margins and a differentiated product.")
        self.pitch_box.delete("1.0", "end")
        self.pitch_box.insert("1.0", sample)
        self.extract_details()

    def extract_details(self):
        pitch = self.pitch_box.get("1.0", "end").strip()
        if not pitch:
            messagebox.showwarning("Pitch required", "Enter a startup pitch first.")
            return
        try:
            self.extracted = self.parser.parse(pitch)
            values = self.prediction_service.prepare_pitch(self.extracted)
            for key, variable in self.pitch_values.items():
                variable.set("" if pd.isna(values[key]) else str(values[key]))
            for key, variable in self.flag_vars.items():
                variable.set(bool(values[key]))
            self.status.config(text="Details extracted. Review or edit the values, then click Analyze pitch.")
        except Exception as exc:
            messagebox.showerror("Extraction error", str(exc))

    def _collect_inputs(self):
        if not self.extracted:
            self.extract_details()
            if not self.extracted:
                raise ValueError("Enter a pitch and extract its details first.")
        pitch_values = self.prediction_service.prepare_pitch(self.extracted)
        for key, variable in self.pitch_values.items():
            value = variable.get().strip()
            if key in ("industry", "state"):
                pitch_values[key] = value or "Unknown"
            else:
                try:
                    pitch_values[key] = float(value) if value else np.nan
                except ValueError:
                    raise ValueError(f"Please enter a valid number for {key.replace('_', ' ')}.")
        for key, variable in self.flag_vars.items():
            pitch_values[key] = int(variable.get())
        if any(np.isfinite(pitch_values[key]) and pitch_values[key] < 0
               for key in ("ask", "valuation", "revenue", "monthly_sales")):
            raise ValueError("Ask, valuation, revenue and monthly sales cannot be negative.")
        if np.isfinite(pitch_values["equity"]) and not 0 < pitch_values["equity"] <= 100:
            raise ValueError("Equity offered must be greater than 0 and at most 100%.")
        if ((np.isfinite(pitch_values["gross_margin"]) and not 0 <= pitch_values["gross_margin"] <= 100) or
                (np.isfinite(pitch_values["net_margin"]) and not -100 <= pitch_values["net_margin"] <= 100)):
            raise ValueError("Gross margin must be 0–100% and net margin must be -100–100%.")
        selected_sharks = [name for name, variable in self.shark_vars.items() if variable.get()]
        if not selected_sharks:
            raise ValueError("Select at least one shark to evaluate.")
        return pitch_values, selected_sharks

    @staticmethod
    def _money(value):
        return f"₹{value:,.1f} lakh"

    ANALYSIS_TASKS = [
        ("Deal Prediction", "deal", "Predicts whether a pitch results in an accepted deal.", "classification"),
        ("Final Deal Valuation", "valuation", "Estimates the final valuation of a completed deal.", "regression"),
        ("Shark Investment Analysis", "shark_investment", "Predicts whether each shark invests in a startup.", "sharks"),
        ("Total Deal Amount", "deal_amount", "Estimates the total capital in an accepted deal.", "regression"),
        ("Total Deal Equity", "deal_equity", "Estimates the equity percentage in an accepted deal.", "regression"),
        ("Founder Acceptance", "founder_acceptance", "Predicts whether founders accept after receiving an offer.", "classification"),
    ]

    @staticmethod
    def _analysis_metric_rows(metrics, task_type):
        if task_type == "classification":
            metric_labels = [
                ("accuracy", "Accuracy"), ("precision", "Precision"),
                ("recall", "Recall"), ("f1", "F1 Score"),
                ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"),
            ]
            return [(label, f"{metrics[key] * 100:.1f}%")
                    for key, label in metric_labels
                    if isinstance(metrics.get(key), (int, float)) and np.isfinite(metrics[key])]
        rows = []
        if isinstance(metrics.get("r2"), (int, float)) and np.isfinite(metrics["r2"]):
            rows.append(("R² Score", f"{metrics['r2']:.3f}"))
        unit = metrics.get("mae_unit", "")
        for key, label in (("mae", "MAE"), ("rmse", "RMSE")):
            value = metrics.get(key)
            if isinstance(value, (int, float)) and np.isfinite(value):
                rows.append((label, f"{value:,.2f} {unit}".strip()))
        return rows

    @staticmethod
    def _classification_takeaways(metrics):
        takeaways = []
        auc = metrics.get("roc_auc")
        if isinstance(auc, (int, float)):
            if auc < 0.5:
                takeaways.append("The model separates the two outcomes poorly on this held-out sample.")
            elif auc < 0.65:
                takeaways.append("The model's ability to rank likely outcomes is limited on this held-out sample.")
            elif auc < 0.75:
                takeaways.append("The model shows moderate ability to rank likely outcomes on this held-out sample.")
            else:
                takeaways.append("The model ranks likely outcomes well on this held-out sample; this does not guarantee future results.")
        counts = metrics.get("class_counts", {})
        negative, positive = counts.get("negative", 0), counts.get("positive", 0)
        if min(negative, positive) > 0 and max(negative, positive) >= 2 * min(negative, positive):
            takeaways.append(f"The test data is imbalanced ({positive} positive and {negative} negative cases), so accuracy alone can be misleading.")
        precision, recall = metrics.get("precision"), metrics.get("recall")
        if isinstance(precision, (int, float)) and isinstance(recall, (int, float)):
            if recall > precision + 0.15:
                takeaways.append("The model finds many positive cases, but a noticeable share of positive predictions are incorrect.")
            elif precision > recall + 0.15:
                takeaways.append("Positive predictions are relatively selective, but some actual positive cases are missed.")
            else:
                takeaways.append("Precision and recall are fairly balanced at the default decision threshold.")
        return takeaways[:3]

    @staticmethod
    def _regression_takeaways(result):
        metrics = result.get("metrics", {})
        diagnostics = result.get("diagnostics", {})
        takeaways = []
        r2 = metrics.get("r2")
        if isinstance(r2, (int, float)):
            if r2 < 0:
                takeaways.append("The model performs worse than predicting a constant average for this held-out sample.")
            elif r2 < 0.4:
                takeaways.append("The model explains a limited share of variation in held-out outcomes.")
            elif r2 < 0.7:
                takeaways.append("The model explains a moderate share of variation in held-out outcomes.")
            else:
                takeaways.append("The model explains much of the variation in this held-out sample; future errors may differ.")
        mae = metrics.get("mae")
        median = diagnostics.get("target_median")
        if isinstance(mae, (int, float)) and isinstance(median, (int, float)) and median > 0:
            ratio = mae / median
            takeaways.append(f"Typical absolute error is about {ratio:.0%} of the test-set median outcome.")
        rmse = metrics.get("rmse")
        if isinstance(mae, (int, float)) and isinstance(rmse, (int, float)) and mae > 0 and rmse > 1.5 * mae:
            takeaways.append("A few larger misses increase the overall error, so estimates are less reliable for some deals.")
        elif isinstance(mae, (int, float)) and isinstance(rmse, (int, float)):
            takeaways.append("MAE is the average absolute error; RMSE gives extra weight to larger misses.")
        return takeaways[:3]

    @staticmethod
    def _make_scrollable_tab(notebook, title):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=title)
        canvas = tk.Canvas(tab, highlightthickness=0, background="#f4f6f8")
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        content = ttk.Frame(canvas, padding=(18, 16))
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window_id, width=event.width))
        return tab, content

    def _add_overview(self, parent, task, result, task_type):
        overview = ttk.LabelFrame(parent, text="Model Overview", padding=(12, 9))
        overview.pack(fill="x", pady=(0, 12))
        if task_type == "sharks":
            shark_results = result.get("metrics", {})
            records = [entry.get("metrics", {}).get("test_rows") for entry in shark_results.values()]
            records = [count for count in records if isinstance(count, int)]
            record_text = (f"Per shark: {min(records)}–{max(records)} held-out records"
                           if records else "Per shark: unavailable")
            selected_model = "Five Individual Shark Prediction Models"
        else:
            record_count = result.get("metrics", {}).get("test_rows")
            record_text = str(record_count) if isinstance(record_count, int) else "Unavailable"
            selected_model = result.get("model") or "Unavailable"

        fields = [
            ("Prediction Task", task[2]),
            ("Selected Model", selected_model),
            ("Evaluation Records", record_text),
            ("Evaluation Method", "Performance measured on held-out test data."),
        ]
        for row, (label, value) in enumerate(fields):
            ttk.Label(overview, text=label, style="AnalysisKey.TLabel").grid(
                row=row, column=0, sticky="nw", padx=(0, 14), pady=3)
            ttk.Label(overview, text=value, wraplength=920, justify="left").grid(
                row=row, column=1, sticky="ew", pady=3)
        overview.columnconfigure(1, weight=1)

    def _add_metric_table(self, parent, metrics, task_type):
        section = ttk.LabelFrame(parent, text="Model Performance (Held-Out Test)", padding=10)
        section.pack(fill="x", pady=(0, 12))
        tree = ttk.Treeview(section, columns=("metric", "performance"), show="headings",
                            height=6, style="Analysis.Treeview")
        tree.heading("metric", text="Metric")
        tree.heading("performance", text="Performance")
        tree.column("metric", width=210, minwidth=150, stretch=True, anchor="w")
        tree.column("performance", width=270, minwidth=180, stretch=True, anchor="w")
        for metric, value in self._analysis_metric_rows(metrics, task_type):
            tree.insert("", "end", values=(metric, value))
        tree.pack(fill="x", expand=True)
        if not tree.get_children():
            ttk.Label(section, text="Held-out performance metrics are not available.").pack(anchor="w", pady=8)

    @staticmethod
    def _add_takeaways(parent, takeaways):
        section = ttk.LabelFrame(parent, text="Key Takeaways", padding=(12, 9))
        section.pack(fill="x")
        message = tk.Label(section, text="\n".join(f"•  {item}" for item in takeaways),
                           background="#ffffff", foreground="#303942", font=("Segoe UI", 10),
                           justify="left", anchor="w", padx=2, pady=2, wraplength=900)
        message.pack(fill="x", anchor="w")
        section.bind("<Configure>", lambda event: message.configure(wraplength=max(320, event.width - 36)))

    def _build_task_tab(self, notebook, task, metadata):
        title, key, _description, task_type = task
        _tab, content = self._make_scrollable_tab(notebook, title)
        result = metadata.get("results", {}).get(key, {})
        self._add_overview(content, task, result, task_type)
        if not result:
            ttk.Label(content, text="Evaluation results are not available. Retrain the models to generate them.").pack(anchor="w")
            return
        self._add_metric_table(content, result.get("metrics", {}), task_type)
        takeaways = (self._classification_takeaways(result.get("metrics", {}))
                     if task_type == "classification" else self._regression_takeaways(result))
        self._add_takeaways(content, takeaways or ["No additional observations are available for this evaluation."])

    def _build_shark_tab(self, notebook, task, metadata):
        title, key, _description, _task_type = task
        _tab, content = self._make_scrollable_tab(notebook, title)
        result = metadata.get("results", {}).get(key, {})
        self._add_overview(content, task, result, "sharks")
        shark_results = result.get("metrics", {})
        section = ttk.LabelFrame(content, text="Individual Shark Performance (Held-Out Test)", padding=10)
        section.pack(fill="x", pady=(0, 12))
        columns = ("shark", "records", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
        tree = ttk.Treeview(section, columns=columns, show="headings",
                    height=5, style="Analysis.Treeview")
        headings = {
            "shark": "Shark", "records": "Test Records", "accuracy": "Accuracy",
            "precision": "Precision", "recall": "Recall", "f1": "F1 Score",
            "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
        }
        widths = {"shark": 115, "records": 105, "accuracy": 105, "precision": 105,
                  "recall": 105, "f1": 100, "roc_auc": 105, "pr_auc": 100}
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], minwidth=80,
                        anchor="w" if column == "shark" else "center", stretch=True)
        for shark in SHARKS:
            shark_result = shark_results.get(shark)
            metrics = shark_result.get("metrics", {}) if shark_result else {}
            values = [shark, metrics.get("test_rows", "")]
            for name in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
                value = metrics.get(name)
                values.append(f"{value * 100:.1f}%" if isinstance(value, (int, float)) and np.isfinite(value) else "")
            tree.insert("", "end", values=values)
        horizontal = ttk.Scrollbar(section, orient="horizontal", command=tree.xview)
        tree.configure(xscrollcommand=horizontal.set)
        tree.pack(fill="x", expand=True)
        horizontal.pack(fill="x")
        if not shark_results:
            ttk.Label(section, text="Shark evaluation results are not available.").pack(anchor="w", pady=8)
        self._add_takeaways(content, self._shark_takeaways(shark_results))

    @staticmethod
    def _shark_takeaways(shark_results):
        evaluated = [(name, item.get("metrics", {})) for name, item in shark_results.items()
                     if isinstance(item.get("metrics", {}).get("roc_auc"), (int, float))]
        if not evaluated:
            return ["Shark-level held-out metrics are not available."]
        strongest = max(evaluated, key=lambda pair: pair[1]["roc_auc"])
        weakest = min(evaluated, key=lambda pair: pair[1]["roc_auc"])
        takeaways = [f"{strongest[0]} has the strongest held-out ranking score ({strongest[1]['roc_auc']:.1%} ROC-AUC)."]
        if weakest[0] != strongest[0]:
            takeaways.append(f"Performance varies by shark; {weakest[0]}'s ROC-AUC is {weakest[1]['roc_auc']:.1%}.")
        positive_rates = []
        for name, metrics in evaluated:
            counts = metrics.get("class_counts", {})
            total = counts.get("positive", 0) + counts.get("negative", 0)
            if total:
                positive_rates.append((name, counts.get("positive", 0) / total))
        if positive_rates:
            low = min(positive_rates, key=lambda pair: pair[1])
            high = max(positive_rates, key=lambda pair: pair[1])
            if high[1] - low[1] >= 0.15:
                takeaways.append("Investment frequency differs across sharks, so classification difficulty is not the same for each model.")
            else:
                takeaways.append("Each row is a separate shark model; scores should be interpreted individually.")
        return takeaways[:3]

    def _close_model_analysis(self):
        if self._model_analysis_window is not None:
            self._model_analysis_window.destroy()
            self._model_analysis_window = None

    def show_model_details(self):
        if self._model_analysis_window is not None and self._model_analysis_window.winfo_exists():
            self._model_analysis_window.deiconify()
            self._model_analysis_window.lift()
            self._model_analysis_window.focus_force()
            return
        try:
            metadata = self.prediction_service.metadata_reader.get_metadata()
            report = tk.Toplevel(self.root)
            self._model_analysis_window = report
            report.title("Model Analysis & Performance")
            report.geometry("1120x760")
            report.minsize(820, 600)
            report.transient(self.root)
            report.protocol("WM_DELETE_WINDOW", self._close_model_analysis)

            style = ttk.Style(report)
            style.configure("AnalysisKey.TLabel", font=("Segoe UI", 10, "bold"), foreground=self.TEXT)
            style.configure("Analysis.TNotebook.Tab", padding=(12, 8), font=("Segoe UI", 9, "bold"))
            style.configure("Analysis.Treeview", font=("Segoe UI", 10), rowheight=27)
            style.configure("Analysis.Treeview.Heading", font=("Segoe UI", 9, "bold"))

            notebook = ttk.Notebook(report, style="Analysis.TNotebook")
            notebook.pack(fill="both", expand=True, padx=14, pady=(14, 8))
            for task in self.ANALYSIS_TASKS:
                if task[3] == "sharks":
                    self._build_shark_tab(notebook, task, metadata)
                else:
                    self._build_task_tab(notebook, task, metadata)
            notebook.select(0)

            footer = ttk.Frame(report, padding=(0, 0, 14, 12))
            footer.pack(fill="x")
            ttk.Button(footer, text="Close", command=self._close_model_analysis).pack(side="right")
        except Exception as exc:
            self._model_analysis_window = None
            messagebox.showerror("Model details error", str(exc))

    def analyze_pitch(self):
        try:
            p, selected = self._collect_inputs()
            result = self.prediction_service.predict(p, selected)
            self.result_box.config(state="normal")
            self.result_box.delete("1.0", "end")
            self.result_box.insert("1.0", self._format_prediction_results(result, p))
            self.result_box.config(state="disabled")
            self.status.config(text="Prediction completed.")
        except Exception as exc:
            messagebox.showerror("Prediction error", str(exc))
            self.status.config(text="Prediction could not be completed.")

    def clear_form(self):
        self.pitch_box.delete("1.0", "end")
        for var in self.pitch_values.values():
            var.set("")
        for var in self.flag_vars.values():
            var.set(False)
        for var in self.shark_vars.values():
            var.set(True)
        self.extracted = {}
        self.result_box.config(state="normal")
        self.result_box.delete("1.0", "end")
        self.result_box.config(state="disabled")
        self.status.config(text="Form cleared.")


def main():
    """Composition root: create concrete dependencies and inject abstractions."""
    root = tk.Tk()
    try:
        store = LocalModelStore(MODEL_DIR)
        # Validate the required files early so the GUI can show a clear error.
        store.get_metadata()
        parser = PitchTextParser()
        service = PitchPredictionService(store, store)
        SharkTankGUI(root, parser, service)
    except Exception as exc:
        root.destroy()
        messagebox.showerror("Application setup error", str(exc))
        return
    root.mainloop()


if __name__ == "__main__":
    main()
