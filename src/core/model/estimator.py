
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt
from scipy.stats import randint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline

from core.constants.project_constants import MODEL_DATA_DIRECTORY
from core.constants.project_constants import TOPIC_CATEGORIES


class Estimator:
    def __init__(self, tfidf_max_features=5000, n_estimators=100, poly_degrees=None):
        self.target_name = 'topic_id'
        self.features = ['message']
        self.n_estimators = n_estimators
        self.poly_degrees = poly_degrees
        self.tfidf_max_features = tfidf_max_features
        self.topics = TOPIC_CATEGORIES
        self.y_pred = None
        self.param_search = None
        self.param_distributions = None
        self.param_grid = None
        self.cv = None
        self.model_pipeline = None
        self.scoring_metric = None

    def set_model(self, word_model=None, classifier=None):

        if word_model is None:
            word_model = TfidfVectorizer(stop_words='english', max_features=self.tfidf_max_features)

        if classifier is None:
            classifier = LogisticRegression(random_state=1506)

        self.scoring_metric = 'f1_weighted'

        self.model_pipeline = Pipeline(steps=[('vec', word_model),
                                              ('classifier', classifier)])

        self.cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1506)

        self.param_grid = dict(dict(vec__max_features=[self.tfidf_max_features]))

        self.param_distributions = dict(classifier__n_estimators=[randint(50, self.n_estimators)])


    def train_test_split(self, df):
        return train_test_split(
            df[self.features],
            df[self.target_name],
            test_size=0.2,
            random_state=1506,
            stratify=df[self.target_name])

    def train_with_grid_search(self, X_train, y_train):
        self.param_search = GridSearchCV(self.model_pipeline,
                                        param_grid=self.param_grid,
                                        cv=self.cv,
                                        scoring=self.scoring_metric)

        self.param_search.fit(X_train, y_train)
        self.model_pipeline = self.param_search.best_estimator_
        # self.y_pred = self.model_pipeline.predict_proba(X_test)

    def train_with_random_search(self, X_train, y_train, n_iter=10):
        self.param_search = RandomizedSearchCV(self.model_pipeline,
                                              param_distributions=self.param_distributions,
                                              cv=self.cv,
                                              scoring=self.scoring_metric,
                                              n_iter=n_iter)

        self.param_search.fit(X_train, y_train)
        self.model_pipeline = self.param_search.best_estimator_
        # self.y_pred = self.model_pipeline.predict_proba(X_test)

    @staticmethod
    def get_metrics(y_train_preds, y_test_preds, y_train, y_test):
        return {'accuracy': [accuracy_score(y_train, np.argmax(y_train_preds, axis=1)),
                             accuracy_score(y_test, np.argmax(y_test_preds, axis=1))],
                'f1_score': [f1_score(y_train, np.argmax(y_train_preds, axis=1), average='weighted'),
                             f1_score(y_test, np.argmax(y_test_preds, axis=1), average='weighted')],
                'precision': [precision_score(y_train, np.argmax(y_train_preds, axis=1), average='weighted'),
                              precision_score(y_test, np.argmax(y_test_preds, axis=1), average='weighted')],
                'recall': [recall_score(y_train, np.argmax(y_train_preds, axis=1), average='weighted'),
                           recall_score(y_test, np.argmax(y_test_preds, axis=1), average='weighted')]
                }

    def print_results_report(self, y_train_preds, y_test_preds, y_train, y_test):

        print("Train Accuracy : {:.3f}".format(accuracy_score(y_train, np.argmax(y_train_preds, axis=1))))
        print("Test  Accuracy : {:.3f}".format(accuracy_score(y_test, np.argmax(y_test_preds, axis=1))))
        print("\nClassification Report : ")
        print(classification_report(y_test, np.argmax(y_test_preds, axis=1), target_names=self.topics))

    def print_results_report_full(self, y_train_preds, y_test_preds, y_train, y_test):

        print("Train Accuracy : {:.3f}".format(accuracy_score(y_train, np.argmax(y_train_preds, axis=1))))
        print("Test  Accuracy : {:.3f}".format(accuracy_score(y_test, np.argmax(y_test_preds, axis=1))))
        print("\nClassification Report : ")
        print(classification_report(y_test, np.argmax(y_test_preds, axis=1), target_names=self.topics))

        skplt.metrics.plot_confusion_matrix(
            [TOPIC_CATEGORIES[i] for i in y_test],
            [TOPIC_CATEGORIES[i] for i in np.argmax(y_test_preds, axis=1)],
            normalize=True,
            title="Confusion Matrix",
            cmap="Purples",
            hide_zeros=True,
            figsize=(5, 5)
        );
        plt.xticks(rotation=90)

    def save_model(self, model_name='model_tf_idf'):
        pickle_file = pathlib.Path.joinpath(MODEL_DATA_DIRECTORY, "{}.pkl".format(model_name))
        with open(pickle_file, 'wb') as file:
            pickle.dump(self.model_pipeline, file)
        print('model saved in {}'.format(pickle_file))

