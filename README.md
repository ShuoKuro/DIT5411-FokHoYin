# DIT5411 Traditional Chinese Handwriting Recognition

A Python and TensorFlow project for the DIT5411 Machine Learning character recognition assignment. The project trains and compares several multilayer classifiers for **13,065 Traditional Chinese character classes**.

The assignment permits either local development or Google Colab. This repository uses a **local-first Apple Silicon workflow**; Colab is optional and is not required.

## Assignment Requirements

- Use the full Traditional Chinese Handwriting Dataset with 13,065 classes.
- Use the first 40 images of each class, in natural filename order, as the training pool and all remaining images as the test set.
- Generate at least 200 training samples per class using image augmentation.
- Train several multilayer TensorFlow models with a 13,065-neuron softmax output layer.
- Evaluate every model on the same original, non-augmented test images.
- Document the augmentation method, model architectures, measured test accuracies, and references.

The repository name follows the required `DIT5411-your-name` format. All project documentation and program output are in English, except for Chinese class labels and source quotations.

## Local Hardware and Verified Backend

The primary development machine has:

- Apple M5 Max with 18 CPU cores and a 40-core GPU
- 128 GB unified memory
- Approximately 522 GB free disk space at the initial environment audit
- Python 3.11.15
- TensorFlow 2.18.1, Keras 3.5.0, and tensorflow-metal 1.2.0

Metal 1.2.0 failed to load with TensorFlow 2.21 in this environment, so the compatible TensorFlow 2.18.1 and Keras 3.5.0 versions are pinned. After installing the compatible versions, TensorFlow successfully detected `/physical_device:GPU:0` and completed forward and backward passes for all three model architectures.

A synthetic batch-256 training-step benchmark produced the following idealized results. These measurements exclude real image reading, decoding, augmentation, validation, and checkpoint overhead. They are performance measurements, **not recognition accuracies**.

| Model | CPU seconds/step | Metal seconds/step |
|---|---:|---:|
| MLP | 0.012 | 0.008 |
| Compact CNN | 0.123 | 0.033 |
| Deeper CNN | 0.570 | 0.129 |

## Dataset

Use the **full 13,065-class, 300×300 dataset**, not the smaller 4,803-class common-character dataset. The official dataset description reports 684,677 images.

Sources:

- [Dataset documentation](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset)
- [Full segmented dataset archive](https://github.com/chenkenanalytic/handwritting_data_all)

Download and combine the segmented archive locally:

```bash
mkdir -p data/source
git clone https://github.com/chenkenanalytic/handwritting_data_all.git data/source/handwritting_data_all
cat data/source/handwritting_data_all/all_data.zip* > data/all_data.zip
```

Extract it with Chinese filename support:

```bash
python -m src.extract_data \
  --archive data/all_data.zip \
  --output-dir data
```

The extractor uses Python 3.11's ZIP metadata decoding with CP950 and checks archive paths before extraction. It does not depend on the Linux-specific `unzip -O big5` option, so the same command works in the local macOS environment. The expected class-directory root is `data/cleaned_data`. Raw data and generated archives are excluded from Git.

The dataset is distributed under **CC BY-NC-SA 4.0**. Retain attribution to AI-FREE Team and the original STUST EECS Chinese MNIST source when sharing this work.

## Local Environment

Create or recreate the Python 3.11 environment and install the pinned dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`tensorflow-metal` has Darwin/ARM64 environment markers. It is installed on this Mac and skipped on non-Apple platforms, including Linux Colab.

Verify the actual compute backend before starting a long run:

```bash
python - <<'PY'
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices())
assert tf.config.list_physical_devices('GPU'), 'No Metal GPU was detected'
PY
```

The Metal check must show both CPU and GPU devices. If GPU detection fails, the training code can still run locally on CPU, but the longer CNN experiments will take more time. No GPU performance should be claimed unless this check and a training step both succeed.

### Optional Colab Alternative

`notebooks/DIT5411_Colab.ipynb` contains the equivalent cloud workflow. It is provided only as a fallback for users without a suitable local machine. Select a GPU runtime in Colab and follow the notebook cells; the Apple-only Metal dependency will be skipped automatically.

## Deterministic Data Split

Create the manifests and audit the class counts:

```bash
python -m src.prepare_data \
  --data-dir data/cleaned_data \
  --output-dir outputs/data \
  --verify-images
```

Preparation stops with an error unless there are exactly 13,065 class directories and every class has at least 41 images. Images are ordered with a natural filename sort, so numeric suffixes are ordered numerically rather than lexicographically.

The official archive stores class directories as numeric IDs. The actual Chinese character is the shared filename prefix, as in `仚_46.png`. Preparation derives the class mapping from these prefixes, validates one prefix per directory and uniqueness across all directories, and assigns label indices in numeric directory order. Single-image prediction therefore returns the Chinese character rather than the numeric directory ID.

| Manifest | Purpose | Images per class |
|---|---|---:|
| `fit.csv` | Development-model training originals | 36 |
| `validation.csv` | Early stopping and learning-rate scheduling | 4 |
| `train.csv` | Complete assignment training pool | 40 |
| `test.csv` | Final recognition evaluation | All remaining images |

The validation images come only from the first 40 training images. Test images are never used for training, augmentation, early stopping, or learning-rate scheduling. Development runs train on `fit.csv`; the final best architecture is retrained on `train.csv` with `--include-validation`.

The data pipeline converts each image to grayscale, resizes it to 64×64 by default, and scales pixel values to `[0,1]`. Integer class labels and sparse categorical cross-entropy avoid allocating a large one-hot label matrix.

## Image Augmentation

Augmentation is applied only to training samples. A deterministic seed is derived from the run seed, relative image path, and variant number, so every sample can be reproduced. No horizontal flipping is used because it can invalidate Chinese characters.

Each affine transformation combines:

| Operation | Range |
|---|---|
| Rotation | -10° to +10° |
| Shear | -0.12 to +0.12 |
| Scaling | 0.90 to 1.10 |
| Horizontal and vertical translation | Up to 5% of image width or height |

Empty border pixels are white. The pipeline uses bilinear interpolation and clips values to `[0,1]`.

For a full 40-image training pool, each original appears once and has four augmented variants, producing **40 + 160 = 200 samples per class**. For a 36-image development fit set, the deterministic schedule adds 164 variants distributed across the originals, again producing exactly 200 samples per class. Consequently, each full training epoch contains **2,613,000 samples**.

Augmented images are generated on demand with `tf.data` rather than stored as 2.613 million additional files. This avoids excessive disk and memory usage while preserving the required number of training samples. Test and validation images are never augmented.

Create a visual and parameter audit for one training image:

```bash
python -m src.augment \
  --image path/to/training_image.png \
  --output outputs/augmentation_preview.png \
  --seed 42
```

## Model Architectures

All models accept a single grayscale image and end with `Dense(13065, activation="softmax")`.

| Model | Architecture | Purpose |
|---|---|---|
| MLP | Flatten → Dense 512 → Dropout → Dense 256 → Dropout → Softmax | Non-convolutional baseline |
| Compact CNN | Three Conv–BatchNorm–ReLU–MaxPool blocks (32, 64, 128 filters) → Global Average Pooling → Dropout → Softmax | Efficient convolutional classifier |
| Deeper CNN | Convolutional stem → four two-convolution residual blocks (32, 64, 128, 256 filters) → Global Average Pooling → Dropout → Softmax | Deeper spatial feature extraction |

The complete layer definitions are in `src/models.py`. Model summaries are saved with each training run.

## Local Training

Run the development experiments locally with the same data, augmentation target, image size, and stopping rules:

```bash
python -m src.train --data-dir data/cleaned_data --model mlp --epochs 20
python -m src.train --data-dir data/cleaned_data --model compact_cnn --epochs 20
python -m src.train --data-dir data/cleaned_data --model deeper_cnn --epochs 20
```

The default float32 policy was used for the successful Metal backend checks. `--mixed-precision` is available as an optional experiment but is not required and should be enabled only after a separate numerical-stability check. Batch size, learning rate, seed, and other settings are configurable through command-line options.

Each run saves:

- `run_config.json`: complete settings, software versions, and physical devices
- `model_summary.txt`: layer architecture and parameter counts
- `history.csv`, `history.json`, and `training_curves.png`: training history
- `best_model.keras`: best validation-loss checkpoint for development runs
- `final_model.keras`: model at the end of the run

After selecting an architecture, retrain it with all 40 original training images per class. Use the best development epoch count as a fixed epoch count:

```bash
python -m src.train \
  --data-dir data/cleaned_data \
  --model deeper_cnn \
  --include-validation \
  --epochs 20 \
  --output-dir outputs/runs/deeper_cnn_full40
```

Replace the model name and epoch count with the selected settings. This full-pool run supplies exactly 200 samples per class per epoch and saves `last_model.keras` after every completed epoch. Because the four development-validation images are returned to training, this run does **not** calculate or report misleading validation metrics. Early stopping is disabled, and recognition performance is measured only on the untouched test set.

## Test Evaluation

Evaluate each development model on `test.csv`. These are the remaining original images for every class, with no augmentation:

```bash
python -m src.evaluate --data-dir data/cleaned_data \
  --model-path outputs/runs/mlp/best_model.keras
python -m src.evaluate --data-dir data/cleaned_data \
  --model-path outputs/runs/compact_cnn/best_model.keras
python -m src.evaluate --data-dir data/cleaned_data \
  --model-path outputs/runs/deeper_cnn/best_model.keras
```

Evaluate the full-pool model separately:

```bash
python -m src.evaluate --data-dir data/cleaned_data \
  --model-path outputs/runs/deeper_cnn_full40/final_model.keras
```

Evaluation streams batches and writes `metrics.json` plus `top5_predictions.csv`. It reports the number of test samples, Top-1 recognition accuracy, Top-5 accuracy, and sparse categorical cross-entropy loss. Full 13,065-class probability matrices are never accumulated in memory.

## Experimental Results

No full-dataset recognition accuracy is claimed yet. Backend checks and synthetic performance benchmarks are complete, but full-dataset training and evaluation remain to be performed. Before submission, replace this section with a comparison table containing each model's measured test loss, Top-1 accuracy, Top-5 accuracy, epoch count, and a link to its run configuration. Use only values from the generated `metrics.json` files and identify the highest measured Top-1 model as the best model.

Smoke-test results from a reduced or synthetic dataset are debugging evidence only and must not be reported as assignment test accuracy.

## Single-Image Prediction

After training and evaluation, run:

```bash
python -m src.predict \
  --model-path outputs/runs/deeper_cnn_full40/final_model.keras \
  --image path/to/handwritten_image.png \
  --manifest-dir outputs/data
```

The command prints the predicted character and ranked class indices and probabilities. Preprocessing is identical to evaluation: grayscale conversion, resizing, and normalization without augmentation.

## Reproducibility and Repository Contents

- `src/extract_data.py`: safe CP950-aware extraction on macOS and other supported platforms
- `src/prepare_data.py`: full-dataset audit and deterministic split creation
- `src/data.py`: manifest handling, exact sample expansion, and `tf.data` pipelines
- `src/augment.py`: reproducible affine augmentation and preview generation
- `src/models.py`: MLP, Compact CNN, and Deeper Residual CNN definitions
- `src/train.py`: configurable training, checkpoints, and history export
- `src/evaluate.py`: streaming test evaluation and Top-5 prediction export
- `src/predict.py`: single-image inference
- `notebooks/DIT5411_Colab.ipynb`: optional GPU-enabled cloud workflow

The `data/`, `outputs/`, virtual-environment, and cache directories are ignored. Do not commit the raw dataset, augmented dataset, model binaries, or large prediction files. Record the actual software and hardware details from `run_config.json` with the final results.

## References

1. AI-FREE Team, [Traditional Chinese Handwriting Dataset](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset), 2020.
2. [Full dataset archive and deployment example](https://github.com/chenkenanalytic/handwritting_data_all).
3. [TensorFlow Keras documentation](https://www.tensorflow.org/api_docs/python/tf/keras).
4. Apple, [TensorFlow Metal Plugin](https://developer.apple.com/metal/tensorflow-plugin/).
5. Darshita Jain, [Introduction to OpenCV](https://github.com/darshitajain/Introduction-to-OpenCV), augmentation reference supplied with the assignment.
6. [Handwritten Digit Recognition using Keras and TensorFlow](https://github.com/kimanalytics/Handwritten-Digit-Recognition-using-Keras-and-TensorFlow), model-documentation reference supplied with the assignment.
