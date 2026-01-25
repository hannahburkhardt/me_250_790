import json
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import altair as alt
import numpy as np
import pandas as pd
import requests
import scipy.stats
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

pd.options.display.max_rows = 100
pd.options.display.max_colwidth = 200
pd.set_option("plotting.backend", "altair")


def load_submission_token() -> Optional[str]:
    """Load SUBMISSION_TOKEN from .env; return None when unavailable."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        print("python-dotenv not installed. Run: pip install python-dotenv")
        return None

    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    load_dotenv(env_path, override=True)

    token = os.getenv("SUBMISSION_TOKEN")
    if token and token != "your_token_here":
        return token

    print("SUBMISSION_TOKEN not found or not set in .env file.")
    print("Please create a .env file with your token to submit responses.")
    print(f"Expected location: {env_path}")
    return None


SUBMISSION_TOKEN = load_submission_token()


def submit_answer(question_num: int, answer: int) -> None:
    """Submit an answer to the course backend."""

    url = "https://bids-class.azurewebsites.net/submit-answer"
    payload = {"token": SUBMISSION_TOKEN, "question_num": question_num, "answer_num": answer}
    response = json.loads(requests.post(url, data=payload).text)

    if response.get("success"):
        print("Correct!" if response.get("correct") else "Incorrect")
    else:
        print(f"Error: {response.get('message')}")

def value_counts_pct(self: pd.Series, raw: bool = False, *args, **kwargs) -> pd.DataFrame:
    counts = self.value_counts(*args, **kwargs).rename("count")
    percents = self.value_counts(normalize=True, *args, **kwargs).rename("percent")
    result = pd.concat([counts, percents], axis=1)

    if raw:
        return result

    formats = {"count": "{:,.0f}", "percent": "{:.2%}"}
    for col, fmt in formats.items():
        result[col] = result[col].apply(lambda val: fmt.format(val))
    return result


pd.Series.value_counts_pct = value_counts_pct
pd.core.groupby.SeriesGroupBy.value_counts_pct = value_counts_pct


def value_counts_pct_all(self: pd.DataFrame, cols: Optional[Sequence[str]] = None, **kwargs) -> pd.DataFrame:
    selected = cols or [c for c in self.columns if self[c].dtype == "object"]
    return pd.concat([self[c].value_counts_pct(**kwargs) for c in selected], keys=selected)


pd.DataFrame.value_counts_pct_all = value_counts_pct_all


def confusion_matrix_chart(y_true: Iterable, y_pred: Iterable) -> alt.Chart:
    cm = confusion_matrix(y_true, y_pred)
    cm_df = (
        pd.DataFrame(cm, columns=["0", "1"], index=["0", "1"])
        .reset_index()
        .melt(id_vars="index")
        .rename(columns={"index": "Actual", "variable": "Predicted"})
    )

    base = alt.Chart(cm_df).encode(x="Predicted", y="Actual")
    return alt.layer(
        base.mark_rect().encode(color="value"),
        base.mark_text().encode(
            text=alt.Text("value", format=",.0f"),
            color=alt.condition(alt.datum.value > cm_df.value.max() / 2, alt.value("white"), alt.value("black")),
        ),
    ).properties(title="Confusion Matrix", width=100)

def auroc_curve_chart(y_true: Iterable, y_score: Iterable) -> alt.Chart:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})

    reference = alt.Chart(pd.DataFrame({"FPR": [0, 1], "TPR": [0, 1]})).mark_line(strokeDash=[5, 5], color="gray").encode(
        x="FPR", y="TPR"
    )

    return (
        alt.Chart(roc_df)
        .mark_line()
        .encode(x="FPR", y="TPR")
        .properties(title=f"ROC Curve (AUC = {roc_auc:.2f})")
        + reference
    )


def auprc_curve_chart(y_true: Iterable, y_score: Iterable) -> alt.Chart:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    pr_df = pd.DataFrame({"Recall": recall, "Precision": precision})

    return (
        alt.Chart(pr_df)
        .mark_line()
        .encode(x="Recall", y="Precision")
        .properties(title=f"PR Curve (AUC = {pr_auc:.2f})")
    )


def mean_confidence_interval(data: Sequence[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    arr = np.asarray(data, dtype=float)
    m, se = float(np.mean(arr)), float(scipy.stats.sem(arr))
    h = se * float(scipy.stats.t.ppf((1 + confidence) / 2.0, len(arr) - 1))
    return m, m - h, m + h


def classification_performance_metrics_table(y_true: Iterable, y_pred: Iterable) -> pd.DataFrame:
    report_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).T
    return report_df.iloc[[0, 1], :]
