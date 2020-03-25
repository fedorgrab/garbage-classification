from tensorflow.keras import layers, models, optimiziers
from tensorflow.keras.applications import VGG16

INPUT_SHAPE_224_224 = (224, 224, 3)


def create_model(input_shape=INPUT_SHAPE_224_224, last_trainable_layers=-3):
    vgg16_base = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

    for layer in vgg16_base.layers[:last_trainable_layers]:
        layer.trainable = False

    model = models.Sequential(
        [
            vgg16_base,
            layers.Flatten(),
            layers.Dense(1024, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(6, activation="softmax"),
        ]
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimiziers.Nadam(lr=1e-4),
        metrics=["acc"],
    )
    return model
