import argparse
import json
from pathlib import Path

import tensorflow as tf

from .augment import load_image


def main():
    parser = argparse.ArgumentParser(description="Predict the Chinese character in one handwritten image.")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--manifest-dir", type=Path, default=Path("outputs/data"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path)
    classes = json.loads((args.manifest_dir / "classes.json").read_text(encoding="utf-8"))
    if model.output_shape[-1] != len(classes):
        raise ValueError("Model output size does not match the class mapping")
    image_size = model.input_shape[1]
    image = load_image(str(args.image), image_size)
    probabilities = model(tf.expand_dims(image, 0), training=False)[0]
    top_k = tf.math.top_k(probabilities, k=min(args.top_k, len(classes)))
    predictions = [
        {
            "rank": rank + 1,
            "class_index": int(label),
            "character": classes[int(label)],
            "probability": float(probability),
        }
        for rank, (label, probability) in enumerate(zip(top_k.indices.numpy(), top_k.values.numpy()))
    ]
    print(
        json.dumps(
            {
                "image": str(args.image),
                "predicted_character": predictions[0]["character"],
                "predictions": predictions,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
