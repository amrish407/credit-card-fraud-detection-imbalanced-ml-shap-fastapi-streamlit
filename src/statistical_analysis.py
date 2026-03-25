from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src import config
from src.logger import get_logger
from src.utils import save_dataframe, save_json

logger = get_logger(__name__)


@dataclass
class StatisticalConfig:
    target_column: str = config.TARGET_COLUMN
    significance_level: float = 0.05
    top_features_for_tests: int = 10


class StatisticalAnalyzer:
    def __init__(self, statistical_config: StatisticalConfig | None = None):
        self.config = statistical_config or StatisticalConfig()

    def descriptive_statistics(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        numeric_df = dataframe.select_dtypes(include=[np.number])
        q1 = numeric_df.quantile(0.25)
        q3 = numeric_df.quantile(0.75)

        summary_df = pd.DataFrame(
            {
                "mean": numeric_df.mean(),
                "median": numeric_df.median(),
                "std": numeric_df.std(ddof=1),
                "variance": numeric_df.var(ddof=1),
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
                "skewness": numeric_df.skew(),
                "kurtosis": numeric_df.kurtosis(),
                "min": numeric_df.min(),
                "max": numeric_df.max(),
            }
        )
        return summary_df.reset_index(names="feature")

    def class_imbalance_summary(self, dataframe: pd.DataFrame) -> dict[str, float]:
        class_counts = dataframe[self.config.target_column].value_counts().to_dict()
        fraud_count = int(class_counts.get(1, 0))
        non_fraud_count = int(class_counts.get(0, 0))
        total_count = fraud_count + non_fraud_count

        fraud_rate = fraud_count / total_count if total_count else 0.0
        non_fraud_rate = non_fraud_count / total_count if total_count else 0.0
        imbalance_ratio = (non_fraud_count / fraud_count) if fraud_count else float("inf")

        return {
            "total_transactions": total_count,
            "fraud_count": fraud_count,
            "non_fraud_count": non_fraud_count,
            "fraud_rate": fraud_rate,
            "non_fraud_rate": non_fraud_rate,
            "imbalance_ratio_non_fraud_to_fraud": imbalance_ratio,
        }

    def outlier_summary_iqr(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        numeric_columns = [col for col in dataframe.columns if col != self.config.target_column]
        records = []

        for column in numeric_columns:
            series = dataframe[column]
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = ((series < lower_bound) | (series > upper_bound)).sum()
            outlier_ratio = outliers / len(series)

            records.append(
                {
                    "feature": column,
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(iqr),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "outlier_count": int(outliers),
                    "outlier_ratio": float(outlier_ratio),
                }
            )

        return pd.DataFrame(records).sort_values("outlier_ratio", ascending=False)

    def correlation_analysis(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        correlations = dataframe.corr(numeric_only=True)[self.config.target_column].drop(self.config.target_column)
        correlation_df = correlations.reset_index()
        correlation_df.columns = ["feature", "correlation_with_class"]
        correlation_df["absolute_correlation"] = correlation_df["correlation_with_class"].abs()
        return correlation_df.sort_values("absolute_correlation", ascending=False)

    def fraud_nonfraud_comparison(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        fraud_df = dataframe[dataframe[self.config.target_column] == 1]
        non_fraud_df = dataframe[dataframe[self.config.target_column] == 0]

        correlation_df = self.correlation_analysis(dataframe)
        selected_features = correlation_df.head(self.config.top_features_for_tests)["feature"].tolist()

        comparison_rows = []
        for feature in selected_features:
            fraud_values = fraud_df[feature]
            non_fraud_values = non_fraud_df[feature]

            t_stat, t_pvalue = stats.ttest_ind(
                fraud_values,
                non_fraud_values,
                equal_var=False,
                nan_policy="omit",
            )
            u_stat, u_pvalue = stats.mannwhitneyu(
                fraud_values,
                non_fraud_values,
                alternative="two-sided",
            )

            comparison_rows.append(
                {
                    "feature": feature,
                    "fraud_mean": float(fraud_values.mean()),
                    "non_fraud_mean": float(non_fraud_values.mean()),
                    "fraud_median": float(fraud_values.median()),
                    "non_fraud_median": float(non_fraud_values.median()),
                    "mean_difference": float(fraud_values.mean() - non_fraud_values.mean()),
                    "t_test_pvalue": float(t_pvalue),
                    "mann_whitney_pvalue": float(u_pvalue),
                    "significant_at_0_05": bool(u_pvalue < self.config.significance_level),
                }
            )

        return pd.DataFrame(comparison_rows)

    def generate_report(self, dataframe: pd.DataFrame) -> dict[str, object]:
        descriptive_df = self.descriptive_statistics(dataframe)
        outlier_df = self.outlier_summary_iqr(dataframe)
        correlation_df = self.correlation_analysis(dataframe)
        comparison_df = self.fraud_nonfraud_comparison(dataframe)
        imbalance_summary = self.class_imbalance_summary(dataframe)

        save_dataframe(descriptive_df, config.METRICS_DIR / "descriptive_statistics.csv")
        save_dataframe(outlier_df, config.METRICS_DIR / "outlier_summary.csv")
        save_dataframe(correlation_df, config.METRICS_DIR / "correlation_summary.csv")
        save_dataframe(comparison_df, config.METRICS_DIR / "fraud_nonfraud_tests.csv")

        summary = {
            "imbalance_summary": imbalance_summary,
            "top_correlated_features": correlation_df.head(10).to_dict(orient="records"),
            "top_outlier_features": outlier_df.head(10).to_dict(orient="records"),
            "hypothesis_test_summary": comparison_df.to_dict(orient="records"),
            "interpretation": {
                "accuracy_warning": "In this highly imbalanced dataset, high accuracy can be achieved by predicting only non-fraud. Precision, Recall, F1, and PR-AUC are more meaningful for fraud detection.",
                "business_tradeoff": "False negatives miss fraud (direct financial loss). False positives trigger manual review and customer friction. Threshold tuning balances this tradeoff.",
            },
        }

        save_json(summary, config.STATISTICS_FILE)
        logger.info("Statistical report saved to %s", config.STATISTICS_FILE)
        return summary
