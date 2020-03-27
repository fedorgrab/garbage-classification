import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from tensorflow.keras import models
from config import ClassifierConfig


class GarbageClassifier:
    same_sample_names: list

    def __init__(self):
        self.model = models.load_model(ClassifierConfig.MODEL_FILE_PATH)

    @staticmethod
    def reshape_image(image):
        return image.reshape(1, *ClassifierConfig.MODEL_IMAGE_SHAPE, 3)

    @staticmethod
    def get_same_sample_names(klass):
        indexes = np.random.randint(
            low=ClassifierConfig.MINIMUM_SAME_SAMPLE_INDEX,
            high=ClassifierConfig.MAXIMUM_SAME_SAMPLE_INDEX,
            size=ClassifierConfig.NUMBER_OF_SAME_SAMPLES,
        )
        return [f"{klass}{index}.jpg" for index in indexes]

    def get_accurate_image(self, image_file):
        image_file_name = image_file.filename
        image_file_path = f"{ClassifierConfig.UPLOADED_IMAGES_FILE_PATH}/{image_file_name}"
        image = imread(image_file)
        imsave(fname=image_file_path, arr=image)
        image_resized = resize(
            image=image,
            output_shape=ClassifierConfig.MODEL_IMAGE_SHAPE,
            anti_aliasing=True
        )
        return self.reshape_image(image_resized), image_file_name

    def classify_image(self, image_file):
        image, saved_image_name = self.get_accurate_image(image_file=image_file)
        prediction_class_index = self.model.predict_classes(x=image)[0]
        return ClassifierConfig.CLASSES[prediction_class_index], saved_image_name
