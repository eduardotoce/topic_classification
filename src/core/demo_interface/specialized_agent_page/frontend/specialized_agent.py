"""
 Chatbot assistant main page definition
"""
import json
import pandas as pd
import streamlit as st
from core.demo_interface.specialized_agent_page.specialized_agent_functions import print_explainer
import requests
from core.constants.project_constants import MODEL_DATA_DIRECTORY
from core.constants.project_constants import TOPIC_CATEGORIES
from core.model.predict import load_model
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components


__all__ = [
    'specialized_agent_main'
]

def specialized_agent_main(sesh):

    if st.session_state['messages']:
        message = st.session_state['messages'][0]

        st.write('The message recorded is: "{}"'.format(message))
        st.write('Here is the customer message analysis :')

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




        plt.figure()
        st.write("SHAP Summary Plot for next 10 instances")
        shap.waterfall_plot(shap_values[0][:, TOPIC_CATEGORIES[preds[0]]], max_display=15, show=False)
        st.pyplot(plt.gcf())  # Render the current Matplotlib figure

        st.write("SHAP Force Plot")
        def st_shap(plot, height=None):
            shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
            components.html(shap_html, height=height)

        st_shap(shap.text_plot(shap_values), height=400)


