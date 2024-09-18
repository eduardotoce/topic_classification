
from core.model.estimator import Estimator
from core.utils.data_modification import read_data
from datetime import datetime

def train():

    model_tf_idf = Estimator()

    model_tf_idf.set_model()

    df, topics = read_data()

    X_train, X_test, y_train, y_test = model_tf_idf.train_test_split(df)

    model_tf_idf.train_with_grid_search(X_train['message'], y_train)

    y_train_preds = model_tf_idf.model_pipeline.predict_proba(X_train.message)
    y_test_preds = model_tf_idf.model_pipeline.predict_proba(X_test.message)

    model_tf_idf.print_results_report_full(y_train_preds, y_test_preds, y_train, y_test)

    current_time = datetime.now()

    model_tf_idf.save_model(model_name='model_{}'.format(current_time.strftime("%Y_%m_%d_%H_%M")))
