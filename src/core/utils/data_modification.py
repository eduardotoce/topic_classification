
import locale

import pandas as pd
import pathlib
from core.constants.project_constants import INPUT_DATA_DIRECTORY

# Load the data from local dataframe

def read_data():
    df = pd.read_csv(pathlib.Path.joinpath(INPUT_DATA_DIRECTORY, "available_conversations.csv"))
    topics = pd.read_csv(pathlib.Path.joinpath(INPUT_DATA_DIRECTORY, "available_topics.csv"))
    return df, topics

