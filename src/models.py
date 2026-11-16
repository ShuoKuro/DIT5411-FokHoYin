import tensorflow as tf


NUM_CLASSES = 13065
MODEL_NAMES = ("mlp", "compact_cnn", "deeper_cnn", "se_cnn")


def build_mlp(num_classes, image_size):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(image_size, image_size, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(256),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32"),
        ],
        name="mlp",
    )


def build_compact_cnn(num_classes, image_size):
    inputs = tf.keras.Input(shape=(image_size, image_size, 1))
    x = inputs
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return tf.keras.Model(inputs, outputs, name="compact_cnn")


def squeeze_excite(inputs, reduction=16):
    channels = inputs.shape[-1]
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Dense(max(channels // reduction, 1), activation="relu")(x)
    x = tf.keras.layers.Dense(channels, activation="sigmoid")(x)
    x = tf.keras.layers.Reshape((1, 1, channels))(x)
    return tf.keras.layers.Multiply()([inputs, x])


def residual_block(inputs, filters, strides=1, use_se=False):
    x = tf.keras.layers.Conv2D(
        filters, 3, strides=strides, padding="same", use_bias=False
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if use_se:
        x = squeeze_excite(x)
    if strides != 1 or inputs.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters, 1, strides=strides, padding="same", use_bias=False
        )(inputs)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    else:
        shortcut = inputs
    x = tf.keras.layers.Add()([x, shortcut])
    return tf.keras.layers.ReLU()(x)


def build_deeper_cnn(num_classes, image_size):
    inputs = tf.keras.Input(shape=(image_size, image_size, 1))
    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    for filters, strides in ((32, 1), (64, 2), (128, 2), (256, 2)):
        x = residual_block(x, filters, strides)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return tf.keras.Model(inputs, outputs, name="deeper_cnn")


def build_se_cnn(num_classes, image_size):
    inputs = tf.keras.Input(shape=(image_size, image_size, 1))
    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    for filters, strides in ((32, 1), (64, 2), (128, 2), (256, 2)):
        x = residual_block(x, filters, strides, use_se=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return tf.keras.Model(inputs, outputs, name="se_cnn")


def build_model(model_name, num_classes=NUM_CLASSES, image_size=64):
    builders = {
        "mlp": build_mlp,
        "compact_cnn": build_compact_cnn,
        "deeper_cnn": build_deeper_cnn,
        "se_cnn": build_se_cnn,
    }
    if model_name not in builders:
        raise ValueError(f"Unknown model {model_name!r}; choose from {MODEL_NAMES}")
    return builders[model_name](num_classes, image_size)
