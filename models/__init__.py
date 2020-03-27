from tensorflow.keras import models


def load_model(filepath):
    return models.load_model(filepath)
