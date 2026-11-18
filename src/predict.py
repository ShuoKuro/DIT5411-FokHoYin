import argparse
import json
import zlib
from pathlib import Path

import tensorflow as tf

from .augment import augment_image, load_image


def main():
    parser = argparse.ArgumentParser(description="Predict the Chinese character in one handwritten image.")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--manifest-dir", type=Path, default=Path("outputs/data"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--tta-views", type=int, default=1)
    parser.add_argument("--tta-profile", choices=("standard", "mild"), default="mild")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    if args.tta_views < 1 or args.top_k < 1:
        raise ValueError("tta_views and top_k must be at least 1")

    model = tf.keras.models.load_model(args.model_path)
    classes = json.loads((args.manifest_dir / "classes.json").read_text(encoding="utf-8"))
    if model.output_shape[-1] != len(classes):
        raise ValueError("Model output size does not match the class mapping")
    image_size = model.input_shape[1]
    image = load_image(str(args.image), image_size)
    if args.data_dir:
        try:
            seed_path = args.image.resolve().relative_to(args.data_dir.resolve()).as_posix()
        except ValueError:
            seed_path = args.image.as_posix()
    else:
        seed_path = args.image.as_posix()

    probabilities = tf.zeros(len(classes), dtype=tf.float32)
    for view in range(args.tta_views):
        if view:
            view_seed = zlib.crc32(f"{args.seed}:{seed_path}:{view}".encode("utf-8")) & 0x7FFFFFFF
            view_image = augment_image(image, [args.seed, view_seed], args.tta_profile)
        else:
            view_image = image
        probabilities += model(tf.expand_dims(view_image, 0), training=False)[0]
    probabilities /= args.tta_views

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
                "tta_views": args.tta_views,
                "tta_profile": args.tta_profile,
                "seed": args.seed,
                "predictions": predictions,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
