import argparse
import json
import platform
from pathlib import Path

import numpy as np
import tensorflow as tf

from .data import make_dataset
from .models import MODEL_NAMES, NUM_CLASSES, build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train one Chinese handwriting recognition model.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--manifest-dir", type=Path, default=Path("outputs/data"))
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--augmentation-target", type=int, default=200)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--shuffle-buffer", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-validation", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    return parser.parse_args()


def save_training_plot(history, output_path):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, metric in zip(axes, ("loss", "accuracy")):
        axis.plot(history[metric], label=f"Training {metric}")
        if f"val_{metric}" in history:
            axis.plot(history[f"val_{metric}"], label=f"Validation {metric}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric.capitalize())
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 16:
        raise ValueError("Epochs and batch size must be positive; image size must be at least 16")
    if args.augmentation_target < 200:
        raise ValueError("The assignment requires at least 200 training samples per class")

    output_dir = args.output_dir or Path("outputs/runs") / (
        args.model + ("_full40" if args.include_validation else "")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_summary = json.loads(
        (args.manifest_dir / "dataset_summary.json").read_text(encoding="utf-8")
    )
    num_classes = dataset_summary["num_classes"]
    if num_classes != NUM_CLASSES:
        raise ValueError(f"Training requires exactly {NUM_CLASSES} classes, found {num_classes}")

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    model = build_model(args.model, num_classes, args.image_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=min(5, num_classes), name="top5_accuracy"
            ),
        ],
    )
    with (output_dir / "model_summary.txt").open("w", encoding="utf-8") as handle:
        model.summary(print_fn=lambda line: handle.write(line + "\n"))

    train_manifest = args.manifest_dir / ("train.csv" if args.include_validation else "fit.csv")
    training = make_dataset(
        train_manifest,
        args.data_dir,
        args.batch_size,
        image_size=args.image_size,
        augment=True,
        samples_per_class=args.augmentation_target,
        shuffle=True,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
    )
    validation = None
    if not args.include_validation:
        validation = make_dataset(
            args.manifest_dir / "validation.csv",
            args.data_dir,
            args.batch_size,
            image_size=args.image_size,
            augment=False,
            shuffle=False,
        )
    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    run_config.update(
        {
            "output_dir": str(output_dir),
            "train_manifest": str(train_manifest),
            "training_samples_per_epoch": training.sample_count,
            "validation_samples": validation.sample_count if validation else 0,
            "validation_returned_to_training": args.include_validation,
            "num_classes": num_classes,
            "tensorflow_version": tf.__version__,
            "python_version": platform.python_version(),
            "physical_devices": [device.name for device in tf.config.list_physical_devices()],
        }
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    callbacks = [
        tf.keras.callbacks.CSVLogger(output_dir / "history.csv"),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    if args.include_validation:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(output_dir / "last_model.keras", save_best_only=False)
        )
    else:
        callbacks.extend(
            [
                tf.keras.callbacks.ModelCheckpoint(
                    output_dir / "best_model.keras", monitor="val_loss", save_best_only=True
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=args.early_stopping_patience,
                    restore_best_weights=True,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6
                ),
            ]
        )

    history = model.fit(
        training.dataset.repeat(),
        epochs=args.epochs,
        steps_per_epoch=training.step_count,
        validation_data=validation.dataset.repeat() if validation else None,
        validation_steps=validation.step_count if validation else None,
        callbacks=callbacks,
        shuffle=False,
    )
    history_values = {
        key: [float(value) for value in values] for key, values in history.history.items()
    }
    (output_dir / "history.json").write_text(
        json.dumps(history_values, indent=2), encoding="utf-8"
    )
    save_training_plot(history_values, output_dir / "training_curves.png")
    model.save(output_dir / "final_model.keras")
    print(
        json.dumps(
            {
                "run_complete": True,
                "model": args.model,
                "output_dir": str(output_dir),
                "training_samples_per_epoch": training.sample_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
