import argparse
import json
from pathlib import Path

import tensorflow as tf


ROTATION_DEGREES = 10.0
SHEAR_FACTOR = 0.12
SCALE_RANGE = (0.90, 1.10)
TRANSLATION_FRACTION = 0.05


def load_image(path, image_size):
    image_bytes = tf.io.read_file(tf.convert_to_tensor(path))
    image = tf.io.decode_image(image_bytes, channels=1, expand_animations=False)
    image.set_shape([None, None, 1])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.clip_by_value(image, 0.0, 1.0)
    return tf.ensure_shape(image, [image_size, image_size, 1])


def sample_augmentation_parameters(seed):
    seed = tf.convert_to_tensor(seed, dtype=tf.int32)

    def uniform(minimum, maximum, offset):
        offset_seed = seed + tf.constant([offset, 0], dtype=tf.int32)
        return tf.random.stateless_uniform([], offset_seed, minimum, maximum)

    return {
        "rotation_degrees": uniform(-ROTATION_DEGREES, ROTATION_DEGREES, 1),
        "shear": uniform(-SHEAR_FACTOR, SHEAR_FACTOR, 2),
        "scale": uniform(SCALE_RANGE[0], SCALE_RANGE[1], 3),
        "translation_x": uniform(-TRANSLATION_FRACTION, TRANSLATION_FRACTION, 4),
        "translation_y": uniform(-TRANSLATION_FRACTION, TRANSLATION_FRACTION, 5),
    }


def augment_image(image, seed):
    parameters = sample_augmentation_parameters(seed)
    height = tf.cast(tf.shape(image)[0], tf.float32)
    width = tf.cast(tf.shape(image)[1], tf.float32)
    center_x = (width - 1.0) / 2.0
    center_y = (height - 1.0) / 2.0

    angle = parameters["rotation_degrees"] * tf.constant(3.141592653589793) / 180.0
    cosine = tf.cos(angle)
    sine = tf.sin(angle)
    scale = parameters["scale"]
    shear = parameters["shear"]
    translation_x = parameters["translation_x"] * width
    translation_y = parameters["translation_y"] * height

    a0 = cosine / scale
    a1 = (sine + shear) / scale
    b0 = -sine / scale
    b1 = cosine / scale
    a2 = center_x - a0 * center_x - a1 * center_y - translation_x
    b2 = center_y - b0 * center_x - b1 * center_y - translation_y
    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])

    transformed = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(image, 0),
        transforms=tf.expand_dims(transform, 0),
        output_shape=tf.shape(image)[:2],
        fill_value=1.0,
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
    )[0]
    transformed = tf.clip_by_value(transformed, 0.0, 1.0)
    return tf.ensure_shape(transformed, image.shape)


def main():
    parser = argparse.ArgumentParser(description="Create an augmentation preview for one image.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    original = load_image(str(args.image), args.image_size)
    images = [original]
    parameters = []
    for variant in range(8):
        variant_seed = [args.seed, variant + 1]
        images.append(augment_image(original, variant_seed))
        values = sample_augmentation_parameters(variant_seed)
        parameters.append({key: float(value.numpy()) for key, value in values.items()})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 3, figsize=(9, 9))
    for index, axis in enumerate(axes.flat):
        axis.imshow(images[index].numpy().squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title("Original" if index == 0 else f"Augmented {index}")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    args.output.with_suffix(".json").write_text(
        json.dumps(parameters, indent=2), encoding="utf-8"
    )
    print(f"Saved augmentation preview to {args.output}")


if __name__ == "__main__":
    main()
