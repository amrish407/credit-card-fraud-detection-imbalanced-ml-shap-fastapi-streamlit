from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.logger import get_logger
from src.utils import save_json

logger = get_logger(__name__)

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency runtime guard
    SHAP_AVAILABLE = False


@dataclass
class ExplainabilityConfig:
    max_background_samples: int = 1200
    max_explain_samples: int = 600
    local_examples: int = 2


class ModelExplainer:
    def __init__(self, explainability_config: ExplainabilityConfig | None = None):
        self.config = explainability_config or ExplainabilityConfig()

    @staticmethod
    def _transform_until_model(model_pipeline, dataframe: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        transformed = dataframe.copy()
        feature_names = transformed.columns.tolist()

        for step_name, step_obj in model_pipeline.steps:
            if step_name == "model":
                break
            if hasattr(step_obj, "transform"):
                transformed = step_obj.transform(transformed)
                if isinstance(transformed, pd.DataFrame):
                    feature_names = transformed.columns.tolist()
                else:
                    transformed = np.asarray(transformed)

        if isinstance(transformed, pd.DataFrame):
            feature_names = transformed.columns.tolist()
            transformed_array = transformed.values
        else:
            transformed_array = np.asarray(transformed)

        return transformed_array, feature_names

    @staticmethod
    def _extract_binary_class_shap_values(shap_output):
        if isinstance(shap_output, list):
            if len(shap_output) == 1:
                return shap_output[0]
            return shap_output[1]

        values = getattr(shap_output, "values", shap_output)
        if isinstance(values, np.ndarray) and values.ndim == 3:
            return values[:, :, 1]
        return values

    @staticmethod
    def _extract_base_value(explainer, shap_output):
        expected_value = getattr(explainer, "expected_value", 0.0)
        if isinstance(expected_value, (list, tuple, np.ndarray)):
            return float(np.asarray(expected_value).reshape(-1)[-1])

        base_values = getattr(shap_output, "base_values", None)
        if base_values is not None:
            base_values_array = np.asarray(base_values)
            if base_values_array.ndim == 2:
                return float(base_values_array[0, -1])
            if base_values_array.ndim == 1:
                return float(base_values_array[0])

        return float(expected_value)

    def generate_shap_artifacts(
        self,
        model_pipeline,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> dict[str, object]:
        if not SHAP_AVAILABLE:
            message = "SHAP is not available. Install shap to generate explainability artifacts."
            logger.warning(message)
            summary = {"status": "skipped", "reason": message}
            save_json(summary, config.SHAP_DIR / "shap_summary.json")
            return summary

        try:
            background_df = X_train.sample(
                n=min(self.config.max_background_samples, len(X_train)),
                random_state=config.RANDOM_STATE,
            )
            explain_df = X_test.sample(
                n=min(self.config.max_explain_samples, len(X_test)),
                random_state=config.RANDOM_STATE,
            )

            transformed_background, feature_names = self._transform_until_model(model_pipeline, background_df)
            transformed_explain, _ = self._transform_until_model(model_pipeline, explain_df)

            final_model = model_pipeline.named_steps["model"]

            try:
                explainer = shap.TreeExplainer(final_model)
                shap_raw = explainer.shap_values(transformed_explain)
            except Exception:
                try:
                    explainer = shap.LinearExplainer(final_model, transformed_background)
                    shap_raw = explainer.shap_values(transformed_explain)
                except Exception:
                    explainer = shap.Explainer(final_model, transformed_background)
                    shap_raw = explainer(transformed_explain)

            shap_values = self._extract_binary_class_shap_values(shap_raw)

            summary_beeswarm_path = config.SHAP_DIR / "shap_summary_beeswarm.png"
            summary_bar_path = config.SHAP_DIR / "shap_summary_bar.png"

            shap.summary_plot(
                shap_values,
                transformed_explain,
                feature_names=feature_names,
                show=False,
                max_display=15,
            )
            plt.tight_layout()
            plt.savefig(summary_beeswarm_path, dpi=160, bbox_inches="tight")
            plt.close()

            shap.summary_plot(
                shap_values,
                transformed_explain,
                feature_names=feature_names,
                show=False,
                max_display=15,
                plot_type="bar",
            )
            plt.tight_layout()
            plt.savefig(summary_bar_path, dpi=160, bbox_inches="tight")
            plt.close()

            probabilities = model_pipeline.predict_proba(explain_df)[:, 1]
            top_indices = np.argsort(probabilities)[::-1][: self.config.local_examples]

            local_paths = []
            base_value = self._extract_base_value(explainer, shap_raw)
            for plot_index, sample_index in enumerate(top_indices, start=1):
                explanation = shap.Explanation(
                    values=shap_values[sample_index],
                    base_values=base_value,
                    data=transformed_explain[sample_index],
                    feature_names=feature_names,
                )
                shap.plots.waterfall(explanation, max_display=12, show=False)
                local_path = config.SHAP_DIR / f"shap_local_waterfall_{plot_index}.png"
                plt.tight_layout()
                plt.savefig(local_path, dpi=160, bbox_inches="tight")
                plt.close()
                local_paths.append(str(local_path))

            summary = {
                "status": "completed",
                "global_beeswarm_plot": str(summary_beeswarm_path),
                "global_bar_plot": str(summary_bar_path),
                "local_plots": local_paths,
                "feature_count": len(feature_names),
            }
            save_json(summary, config.SHAP_DIR / "shap_summary.json")
            logger.info("SHAP artifacts created in %s", config.SHAP_DIR)
            return summary
        except Exception as error:
            logger.exception("Failed to generate SHAP artifacts: %s", error)
            summary = {"status": "failed", "reason": str(error)}
            save_json(summary, config.SHAP_DIR / "shap_summary.json")
            return summary
