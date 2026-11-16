import argparse
import csv
import json
from pathlib import Path

import tensorflow as tf

from .data import make_tta_dataset, read_manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model with deterministic test-time augmentation.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--manifest-dir", type=Path, default=Path("outputs/data"))
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--profile", choices=("standard", "mild"), default="mild")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.views < 1:
        raise ValueError("views must be at least 1")
    output_dir = args.output_dir or args.model_path.parent / (
        f"{args.split}_tta_views{args.views}_{args.profile}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(args.model_path)
    image_size = model.input_shape[1]
    classes = json.loads((args.manifest_dir / "classes.json").read_text(encoding="utf-8"))
    if model.output_shape[-1] != len(classes):
        raise ValueError("Model output size does not match the class mapping")
    manifest = args.manifest_dir / f"{args.split}.csv"
    rows = read_manifest(manifest)
    testing = make_tta_dataset(
        manifest,
        args.data_dir,
        args.batch_size,
        image_size=image_size,
        views=args.views,
        profile=args.profile,
        seed=args.seed,
    )

    total = 0
    top1_correct = 0
    top5_correct = 0
    loss_sum = 0.0
    row_position = 0
    prediction_path = output_dir / "top5_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "path",
                "true_label",
                "true_character",
                "predicted_label",
                "predicted_character",
                "correct",
                "top5_labels",
                "top5_characters",
                "top5_probabilities",
            ),
        )
        writer.writeheader()
        for images, labels in testing.dataset:
            view_probabilities = model(images, training=False)
            view_probabilities = tf.reshape(
                view_probabilities, (-1, args.views, len(classes))
            )
            probabilities = tf.reduce_mean(view_probabilities, axis=1)
            labels = tf.reshape(labels, (-1, args.views))
            if not bool(tf.reduce_all(labels == labels[:, :1])):
                raise RuntimeError("TTA views are not contiguous or labels do not match")
            labels = labels[:, 0]
            losses = tf.keras.losses.sparse_categorical_crossentropy(labels, probabilities)
            top_k = tf.math.top_k(probabilities, k=min(5, len(classes)))
            batch_size = int(labels.shape[0])
            loss_sum += float(tf.reduce_sum(losses).numpy())
            for batch_index in range(batch_size):
                row = rows[row_position]
                row_position += 1
                true_label = int(labels[batch_index].numpy())
                if int(row["label"]) != true_label:
                    raise RuntimeError("Prediction order no longer matches the manifest")
                predicted_label = int(top_k.indices[batch_index, 0].numpy())
                top5_labels = [int(value) for value in top_k.indices[batch_index].numpy()]
                top5_probabilities = [float(value) for value in top_k.values[batch_index].numpy()]
                correct = predicted_label == true_label
                top1_correct += int(correct)
                top5_correct += int(true_label in top5_labels)
                writer.writerow(
                    {
                        "path": row["path"],
                        "true_label": true_label,
                        "true_character": classes[true_label],
                        "predicted_label": predicted_label,
                        "predicted_character": classes[predicted_label],
                        "correct": correct,
                        "top5_labels": ";".join(map(str, top5_labels)),
                        "top5_characters": ";".join(classes[label] for label in top5_labels),
                        "top5_probabilities": ";".join(f"{value:.8f}" for value in top5_probabilities),
                    }
                )
            total += batch_size

    if total != len(rows) or row_position != len(rows):
        raise RuntimeError(f"Expected {len(rows)} predictions, produced {total}")
    metrics = {
        "model_path": str(args.model_path),
        "split": args.split,
        "manifest": str(manifest),
        "samples": total,
        "views_per_sample": args.views,
        "augmented_views_per_sample": args.views - 1,
        "profile": args.profile,
        "seed": args.seed,
        "top1_accuracy": top1_correct / total,
        "top5_accuracy": top5_correct / total,
        "loss": loss_sum / total,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
