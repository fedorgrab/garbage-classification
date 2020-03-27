from tensorflow.keras import layers, models, optimizers

INPUT_SHAPE_384_512 = (384, 512, 3)


def create_model(input_shape):
    model = models.Sequential(
        [
            layers.Convolution2D(
                filters=16,
                input_shape=input_shape,
                kernel_size=(4, 4),
                padding="same",
                activation="relu",
            ),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Convolution2D(
                filters=32, kernel_size=(4, 4), padding="same", activation="relu"
            ),
            layers.MaxPooling2D(input_shape=(2, 2)),
            layers.Convolution2D(
                filters=64, kernel_size=(3, 3), padding="same", activation="relu"
            ),
            layers.Flatten(),
            layers.Dense(units=256, activation="relu"),
            layers.Dense(units=6, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=optimizers.RMSprop(lr=2e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["acc"],
    )
    return model
