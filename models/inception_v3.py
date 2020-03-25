from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers, models, optimizers

INPUT_SHAPE_300_300 = (300, 300, 3)


def create_model(input_shape=INPUT_SHAPE_300_300, weights=None):
    if weights is not None:
        inception_base = InceptionV3(
            weights=None, include_top=False, input_shape=input_shape
        )
        inception_base.load_weights(weights)
    else:
        inception_base = InceptionV3(
            weights="imagenet", include_top=False, input_shape=input_shape
        )

    inception_base.trainable = False
    model = models.Sequential(
        [
            inception_base,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(1024, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(6, activation="softmax"),
        ]
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.nadam(lr=0.001),
        metrics=["accuracy"],
    )
    return model
