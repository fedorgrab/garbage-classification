from tensorflow import keras
from tensorflow.keras import layers, models

INPUT_SHAPE_384_512 = (384, 512, 3)


def create_model(input_shape=INPUT_SHAPE_384_512):
    model = models.Sequential(
        [
            layers.Convolution2D(
                64,
                3,
                strides=2,
                padding="same",
                activation="relu",
                input_shape=input_shape,
            ),
            layers.Convolution2D(64, 3, strides=2, activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Dropout(0.2),
            layers.Convolution2D(128, 3, strides=2, padding="same", activation="relu"),
            layers.Convolution2D(128, 3, strides=2, activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Dropout(0.2),
            layers.Convolution2D(256, 3, strides=2, padding="same", activation="relu"),
            layers.Convolution2D(256, 3, strides=2, padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(2048, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(6, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=keras.losses.categorical_crossentropy,
        metrics=["accuracy"],
    )

    return model
