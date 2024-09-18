"""
Chatbot assistant functions to apply in page "User page"
"""

import requests
from core.constants.project_constants import MODEL_DATA_DIRECTORY
from core.constants.project_constants import TOPIC_CATEGORIES
from core.model.predict import predict_one_instance

__all__ = [
    'create_user_response',
    'chatbot_response',
]

def create_user_response(topic):
    if topic == 'other':
        return 'This is not a supported topic, sorry'
    else:
        return 'Hi you are asking about {}, I will send to to a {} agent'.format(topic, topic)


# Function to simulate chatbot response
def chatbot_response(user_message, model_name='model_best'):
    topic = predict_one_instance(user_message, model_name=model_name)
    return create_user_response(topic)
