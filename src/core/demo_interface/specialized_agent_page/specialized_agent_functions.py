"""
Chatbot assistant functions to apply in page "User page"
"""

from core.constants.project_constants import MODEL_DATA_DIRECTORY
from core.constants.project_constants import TOPIC_CATEGORIES
from core.model.predict import load_model
import shap
import pandas as pd

__all__ = [
    'print_explainer',
]


def print_explainer(message):
    shap.initjs()
    X_message = pd.Series(message)
    model = load_model(model_name='model_best')

    def make_predictions(X):
        return model.predict_proba(X)

    masker = shap.maskers.Text(tokenizer=r"\W+")
    explainer = shap.Explainer(make_predictions, masker=masker, output_names=TOPIC_CATEGORIES)

    preds_proba = model.predict_proba(X_message)
    preds = preds_proba.argmax(axis=1)

    shap_values = explainer(X_message)

    shap.text_plot(shap_values)
    shap.waterfall_plot(shap_values[0][:, TOPIC_CATEGORIES[preds[0]]], max_display=15)
