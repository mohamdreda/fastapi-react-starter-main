"""Post-imputation validation utilities (numeric drift, categorical shift, text artefacts)."""
from typing import Dict, Any
import pandas as pd
from scipy import stats
import seaborn as sns  # noqa: F401  # used for density plots
import matplotlib.pyplot as plt
import os
from .utils import get_artifact_path, timestamped_name


def validate_and_save(before: pd.DataFrame, after: pd.DataFrame, dataset_id: int, user_id: int) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "numeric": {},
        "categorical": {},
        "text": {},
    }

    for col in before.columns:
        if before[col].dtype.kind in "if" and after[col].notna().any():
            ks_stat, ks_p = stats.ks_2samp(before[col].dropna(), after[col].dropna())
            report["numeric"][col] = {"ks_stat": ks_stat, "p": ks_p}

            # KDE plot
            fig, ax = plt.subplots()
            sns.kdeplot(before[col].dropna(), label="before", ax=ax)
            sns.kdeplot(after[col].dropna(), label="after", ax=ax)
            ax.set_title(f"Distribution change: {col}")
            ax.legend()
            plot_name = timestamped_name(f"numeric_drift_{col}", "png")
            fig.savefig(get_artifact_path(dataset_id, user_id, plot_name))
            plt.close(fig)
        elif before[col].dtype == object:
            before_counts = before[col].value_counts(normalize=True)
            after_counts = after[col].value_counts(normalize=True)
            common_idx = before_counts.index.intersection(after_counts.index)
            if len(common_idx) > 1:
                chi2, p, _, _ = stats.chi2_contingency(
                    [before_counts[common_idx], after_counts[common_idx]]
                )
                report["categorical"][col] = {"chi2": chi2, "p": p}

            # text artefact: count of constant token "MANQUANT"
            missing_token = "MANQUANT"
            if (after[col] == missing_token).any():
                report["text"][col] = {"missing_token_count": int((after[col] == missing_token).sum())}

    # Save JSON report
    import json  # local import to avoid overhead when not needed

    json_name = timestamped_name("validation", "json")
    json_path = get_artifact_path(dataset_id, user_id, json_name)
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    return {"path": json_path, "report": report}
