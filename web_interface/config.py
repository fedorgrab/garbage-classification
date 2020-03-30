from os import path

BASE_DIR = path.dirname(__file__)


class Config:
    SECRET_KEY = "you-will-never-guess"
    BASE_DIR = BASE_DIR


class ClassifierConfig:
    MODEL_DIR = path.abspath(path.join(Config.BASE_DIR, "..", "..", "garbage_models"))
    MODEL_NAME = "VGG16_garbage_classifier_2.h5"
    MODEL_FILE_PATH = path.join(MODEL_DIR, MODEL_NAME)
    UPLOADED_IMAGES_FILE_PATH = path.abspath(
        path.join(BASE_DIR, "..", "..", "test-images")
    )
    CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    MODEL_IMAGE_SHAPE = (256, 341)
    MINIMUM_SAME_SAMPLE_INDEX = 1
    MAXIMUM_SAME_SAMPLE_INDEX = 15
    NUMBER_OF_SAME_SAMPLES = 6
