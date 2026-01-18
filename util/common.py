import numpy as np
import altair as alt
import pandas as pd
import os
from pathlib import Path

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

pd.options.display.max_rows = 100
pd.options.display.max_colwidth = 200
pd.set_option("plotting.backend", "altair")

# Load environment variables from .env file
def load_submission_token():
    """
    Load the SUBMISSION_TOKEN from .env file.
    Returns None if not found or if python-dotenv is not installed.
    """
    try:
        from dotenv import load_dotenv
        # Find .env file in the project root (go up from util/ to project root)
        project_root = Path(__file__).parent.parent
        env_path = project_root / '.env'
        load_dotenv(env_path)
        token = os.getenv('SUBMISSION_TOKEN')
        if token and token != 'your_token_here':
            return token
        else:
            print("⚠️  SUBMISSION_TOKEN not found or not set in .env file.")
            print("    Please create a .env file with your token to submit assignments.")
            print(f"    Expected location: {env_path}")
            return None
    except ImportError:
        print("⚠️  python-dotenv not installed. Run: pip install python-dotenv")
        return None

# Automatically load token when the module is imported
SUBMISSION_TOKEN = load_submission_token()

import requests
import json

def submit_answer(question_num, answer):
    url = "https://bids-class.azurewebsites.net/submit-answer"
    data = {"token": SUBMISSION_TOKEN, "question_num": question_num, "answer_num": answer}
    x = requests.post(url, data=data)
    response = json.loads(x.text)
    if response["success"]:
        print("✅ Correct!" if response["correct"] else "❌ Incorrect")
    else:
        print(f"⚠️ Error: {response['message']}")

def value_counts_pct(self: pd.Series, raw=False, *args, **kwargs):
    t = pd.concat([
        self.value_counts(*args, **kwargs).rename("count"), 
        self.value_counts(normalize=True, *args, **kwargs).rename("percent")
    ], axis=1)
    if raw:
        return t
    else:
        formats = {"count": "{:,.0f}", "percent": "{:.2%}"}
        for c, f in formats.items():
            t[c] = t[c].apply(lambda v: f.format(v))
        return t
    
pd.Series.value_counts_pct = value_counts_pct
pd.core.groupby.SeriesGroupBy.value_counts_pct = value_counts_pct

def value_counts_pct_all(self: pd.DataFrame, cols = None, **kwargs):
    if cols is None:
        cols = [c for c in self.columns if self[c].dtype == 'object']
    return pd.concat([self[c].value_counts_pct(**kwargs) for c in cols], keys=cols)

pd.DataFrame.value_counts_pct_all = value_counts_pct_all


def confusion_matrix_chart(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, columns=['0', '1'], index=['0', '1']).reset_index().melt(id_vars='index').rename(columns={'index': 'Actual', 'variable': 'Predicted'})
    base = alt.Chart(cm_df).encode(
        x='Predicted',
        y='Actual'
    )
    return alt.layer(
        base.mark_rect().encode(color='value'),
        base.mark_text().encode(
            text=alt.Text('value', format=",.0f"), 
            color=alt.condition(alt.datum.value > cm_df.value.max()/2, alt.value('white'), alt.value('black'))
        )
    ).properties(
        title='Confusion Matrix',
        width=100,
    )

def auroc_curve_chart(y_test, y_score):
    # implementation similar to RocCurveDisplay with altair
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
    return alt.Chart(roc_df).mark_line().encode(
        x='FPR',
        y='TPR'
    ).properties(
        title=f'ROC Curve (AUC = {roc_auc:.2f})'
    ) + alt.Chart(pd.DataFrame({'FPR': [0, 1], 'TPR': [0, 1]})).mark_line(strokeDash=[5,5], color='gray').encode(
        x='FPR',
        y='TPR'
    )

def auprc_curve_chart(y_test, y_score):
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    pr_auc = average_precision_score(y_test, y_score)
    pr_df = pd.DataFrame({'Recall': recall, 'Precision': precision})
    return alt.Chart(pr_df).mark_line().encode(
        x='Recall',
        y='Precision'
    ).properties(
        title=f'PR Curve (AUC = {pr_auc:.2f})'
    )

import numpy as np
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

from sklearn.metrics import classification_report

def classification_performance_metrics_table(y_true, y_pred) -> pd.DataFrame:
    return pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).T.iloc[[0,1],:]