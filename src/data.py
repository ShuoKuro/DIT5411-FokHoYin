import csv
import json
import math
import re
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf

from .augment import augment_image, load_image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
MANIFEST_FIELDS = ("path", "label", "character", "source_index")


@dataclass
class DatasetBundle:
    dataset: tf.data.Dataset
    sample_count: int
    step_count: int


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def read_manifest(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_manifests(
    data_dir,
    output_dir,
    train_per_class=40,
    validation_per_class=4,
    expected_classes=13065,
    verify_images=False,
):
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    if not 0 < validation_per_class < train_per_class:
        raise ValueError("validation_per_class must be between 1 and train_per_class - 1")

    class_directories = sorted(
        (path for path in data_dir.iterdir() if path.is_dir()),
        key=natural_key,
    )
    if len(class_directories) != expected_classes:
        raise ValueError(
            f"Expected {expected_classes} class directories, found {len(class_directories)} in {data_dir}"
        )

    manifests = {"fit": [], "validation": [], "train": [], "test": []}
    class_counts = []
    class_characters = []
    for label, class_directory in enumerate(class_directories):
        images = sorted(
            (
                path
                for path in class_directory.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=natural_key,
        )
        if len(images) < train_per_class + 1:
            raise ValueError(
                f"Class {class_directory.name!r} has {len(images)} images; "
                f"at least {train_per_class + 1} are required"
            )
        characters = {image_path.stem.rsplit("_", 1)[0] for image_path in images}
        if len(characters) != 1 or not next(iter(characters)):
            raise ValueError(
                f"Class directory {class_directory.name!r} must use one shared Chinese-character filename prefix"
            )
        character = characters.pop()
        class_characters.append(character)
        if verify_images:
            from PIL import Image

            for image_path in images:
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except Exception as error:
                    raise ValueError(f"Invalid image: {image_path}: {error}") from error

        rows = [
            {
                "path": image_path.relative_to(data_dir).as_posix(),
                "label": label,
                "character": character,
                "source_index": source_index,
            }
            for source_index, image_path in enumerate(images)
        ]
        train_rows = rows[:train_per_class]
        validation_rows = train_rows[-validation_per_class:]
        fit_rows = train_rows[:-validation_per_class]
        test_rows = rows[train_per_class:]
        manifests["fit"].extend(fit_rows)
        manifests["validation"].extend(validation_rows)
        manifests["train"].extend(train_rows)
        manifests["test"].extend(test_rows)
        class_counts.append(
            {
                "label": label,
                "character": character,
                "fit": len(fit_rows),
                "validation": len(validation_rows),
                "train": len(train_rows),
                "test": len(test_rows),
                "total": len(rows),
            }
        )

    if len(set(class_characters)) != len(class_characters):
        raise ValueError("Chinese-character filename prefixes are not unique across class directories")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in manifests.items():
        write_manifest(output_dir / f"{split}.csv", rows)
    with (output_dir / "class_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("label", "character", "fit", "validation", "train", "test", "total")
        )
        writer.writeheader()
        writer.writerows(class_counts)

    classes = class_characters
    (output_dir / "classes.json").write_text(
        json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "class_to_index.json").write_text(
        json.dumps({name: index for index, name in enumerate(classes)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "num_classes": len(classes),
        "train_per_class": train_per_class,
        "fit_per_class": train_per_class - validation_per_class,
        "validation_per_class": validation_per_class,
        "augmented_samples_per_class": 200,
        "fit_original_samples": len(manifests["fit"]),
        "validation_samples": len(manifests["validation"]),
        "train_pool_samples": len(manifests["train"]),
        "test_samples": len(manifests["test"]),
        "total_samples": sum(count["total"] for count in class_counts),
        "split_rule": (
            f"Natural filename order; first {train_per_class} form the training pool "
            "and all remaining images form the test set."
        ),
        "validation_rule": (
            f"The last {validation_per_class} images within each first-{train_per_class} "
            "training pool are held out for development runs."
        ),
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def make_dataset(
    manifest_path,
    data_dir,
    batch_size,
    image_size=64,
    augment=False,
    samples_per_class=200,
    shuffle=False,
    seed=42,
    shuffle_buffer=50000,
):
    manifest_path = Path(manifest_path)
    data_dir = Path(data_dir).resolve()
    rows = read_manifest(manifest_path)
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    counts = Counter(int(row["label"]) for row in rows)
    if augment and (samples_per_class < max(counts.values()) or len(set(counts.values())) != 1):
        raise ValueError("Augmented samples per class must be at least the equal original class count")
    sample_count = samples_per_class * len(counts) if augment else len(rows)

    def entry_generator():
        positions = Counter()
        for row in rows:
            label = int(row["label"])
            position = positions[label]
            positions[label] += 1
            repeat_count = 1
            if augment:
                base, remainder = divmod(samples_per_class, counts[label])
                repeat_count = base + int(position < remainder)
            relative_path = row["path"]
            absolute_path = str(data_dir / Path(relative_path))
            path_seed = zlib.crc32(f"{seed}:{relative_path}".encode("utf-8")) & 0x7FFFFFFF
            yield absolute_path, label, -1, path_seed
            for variant in range(repeat_count - 1):
                variant_seed = zlib.crc32(
                    f"{seed}:{relative_path}:{variant}".encode("utf-8")
                ) & 0x7FFFFFFF
                yield absolute_path, label, variant, variant_seed

    output_signature = (
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    dataset = tf.data.Dataset.from_generator(entry_generator, output_signature=output_signature)
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(sample_count, shuffle_buffer),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def process(path, label, variant, entry_seed):
        image = load_image(path, image_size)
        image = tf.cond(
            variant >= 0,
            lambda: augment_image(image, [seed, entry_seed]),
            lambda: image,
        )
        return image, label

    dataset = dataset.map(process, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    dataset = dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return DatasetBundle(dataset, sample_count, math.ceil(sample_count / batch_size))
