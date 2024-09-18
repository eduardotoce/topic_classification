import numpy as np
import pandas as pd
import pathlib
import pickle
import scikitplot as skplt
from core.constants.project_constants import MODEL_DATA_DIRECTORY
from core.constants.project_constants import TOPIC_CATEGORIES
from core.utils.data_modification import read_data
from sklearn.metrics import accuracy_score


def load_model(model_name='model_best'):
    with open(pathlib.Path.joinpath(MODEL_DATA_DIRECTORY, "{}.pkl".format(model_name)), 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

def predict(model_name='model_best'):
    df, topics = read_data()
    model = load_model(model_name)
    y_test_preds = model.predict_proba(df.message)
    return y_test_preds

def predict_one_instance(message, model_name='model_best'):
    model = load_model(model_name)
    X_one_instance = pd.Series(message)
    proba_one_instance = model.predict_proba(X_one_instance)
    return TOPIC_CATEGORIES[proba_one_instance.argmax(axis=1)[0]]

def print_predictions_report(y_test, y_test_preds):
    # print("Train Accuracy : {:.3f}".format(accuracy_score(y_train, np.argmax(y_train_preds, axis=1))))
    print("Test  Accuracy : {:.3f}".format(accuracy_score(y_test, np.argmax(y_test_preds, axis=1))))
    print("\nClassification Report : ")
    print(classification_report(y_test, np.argmax(test_preds, axis=1), target_names=selected_categories))

    skplt.metrics.plot_confusion_matrix([TOPIC_CATEGORIES[0] for i in y_test],
                                        [TOPIC_CATEGORIES[0] for i in np.argmax(y_test_preds, axis=1)],
                                        normalize=True,
                                        title="Confusion Matrix",
                                        cmap="Purples",
                                        hide_zeros=True,
                                        figsize=(5,5)
                                        )

    plt.xticks(rotation=90)



def print_explainer_2(message, loaded_model):
    shap.initjs()
    X_manual = pd.Series(message)

    def make_predictions(X):
        preds = loaded_model.predict_proba(X)
        return preds

    masker = shap.maskers.Text(tokenizer=r"\W+")
    explainer = shap.Explainer(make_predictions, masker=masker, output_names=selected_categories)

    shap_values = explainer(X_manual)

    shap.text_plot(shap_values)
    shap.waterfall_plot(shap_values[0][:, selected_categories[preds[0]]], max_display=15)

