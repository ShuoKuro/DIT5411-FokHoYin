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

A synthetic batch-256 training-step benchmark produced the following idealized Metal results. These measurements exclude real image reading, decoding, augmentation, validation, and checkpoint overhead.

| Model | Metal seconds/step |
|---|---:|
| MLP with LeakyReLU | 0.009 |
| Compact CNN | 0.033 |
| Deeper CNN | 0.129 |

A second 20-step Metal benchmark used the real `fit.csv` images and deterministic augmentation pipeline at batch size 256. It included image reading, decoding, resizing, and augmentation, but excluded validation and checkpoint overhead.

| Model | Real-data Metal seconds/step | Estimated hours per full epoch |
|---|---:|---:|
| MLP with LeakyReLU | 0.033 | 0.09 |
| Compact CNN | 0.037 | 0.10 |
| Deeper CNN | 0.126 | 0.36 |

All benchmark figures are performance measurements, **not recognition accuracies**.

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

Preparation requires exactly 13,065 class directories and verifies every image when `--verify-images` is used. Images are ordered with a natural filename sort, so numeric suffixes are ordered numerically rather than lexicographically.

The official archive stores class directories as numeric IDs. The actual Chinese character is the shared filename prefix, as in `仚_46.png`. Preparation derives the class mapping from these prefixes, validates one prefix per directory and uniqueness across all directories, and assigns label indices in numeric directory order. Single-image prediction therefore returns the Chinese character rather than the numeric directory ID.

| Manifest | Purpose | Images per class |
|---|---|---:|
| `fit.csv` | Development-model training originals | 36 for standard classes; fewer for short classes |
| `validation.csv` | Early stopping and learning-rate scheduling | Up to 4 |
| `train.csv` | Complete assignment training pool | First 40 for standard classes; approximately first 80% for short classes |
| `test.csv` | Final recognition evaluation | All remaining original images |

The official data contains **43 classes with only 16–40 images**, so the requested 40 training images plus a nonempty test set is mathematically impossible for those classes. Standard classes with more than 40 images use exactly the first 40 for training. Each short class uses approximately its first 80% for training, always retaining at least one original image for testing. This necessary exception preserves all 13,065 classes and never duplicates test images into training. Exact counts for every short class are recorded in `outputs/data/dataset_summary.json`; all classes are listed in `class_counts.csv`.

The completed local audit successfully verified all **684,677 images** and 13,065 unique character labels. The resulting manifests contain 469,620 development-fit originals, 52,260 validation originals, **521,880 training-pool originals**, and **162,797 original test images**. Train and test paths are disjoint, and their counts sum to the full dataset.

Validation images come only from each training pool. Test images are never used for training, augmentation, early stopping, or learning-rate scheduling. Development runs train on `fit.csv`; the final best architecture is retrained on `train.csv` with `--include-validation`.

The data pipeline converts each image to grayscale, resizes it to 64×64 by default, and scales pixel values to `[0,1]`. Integer class labels and sparse categorical cross-entropy avoid allocating a large one-hot label matrix.

## Image Augmentation

Augmentation is applied only to training samples. A deterministic seed is derived from the run seed, relative image path, and variant number, so every sample can be reproduced. No horizontal flipping is used because it can invalidate Chinese characters.

Each affine transformation combines:

| Operation | Standard profile | Mild profile |
|---|---|---|
| Rotation | -10° to +10° | -7° to +7° |
| Shear | -0.12 to +0.12 | -0.08 to +0.08 |
| Scaling | 0.90 to 1.10 | 0.93 to 1.07 |
| Horizontal and vertical translation | Up to 5% of image width or height | Up to 3% |

Select the profile with `--augmentation-profile standard` or `--augmentation-profile mild`. Empty border pixels are white. The pipeline uses bilinear interpolation and clips values to `[0,1]`.

For a full 40-image training pool, each original appears once and has four augmented variants, producing **40 + 160 = 200 samples per class**. For a 36-image development fit set, the deterministic schedule adds 164 variants distributed across the originals, again producing exactly 200 samples per class. Short classes receive enough deterministic variants to reach the same target without using test images. Consequently, each full training epoch contains **2,613,000 samples**.

Augmented images are generated on demand with `tf.data` rather than stored as 2.613 million additional files. This avoids excessive disk and memory usage while preserving the required number of training samples. Test and validation images are never augmented.

Two training modes are available. The default `--augmentation-mode fixed` repeats the same variants across epochs for baseline compatibility. `--augmentation-mode per_epoch` lazily concatenates epoch-specific datasets and includes the actual epoch number in the deterministic seed, providing different variants each epoch while preserving reproducibility, resume behavior, and exactly 200 samples per class.

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
| MLP | Flatten → Dense 512 → LeakyReLU 0.1 → Dropout → Dense 256 → LeakyReLU 0.1 → Dropout → Softmax | Non-convolutional baseline |
| Compact CNN | Three Conv–BatchNorm–ReLU–MaxPool blocks (32, 64, 128 filters) → Global Average Pooling → Dropout → Softmax | Efficient convolutional classifier |
| Deeper CNN | Convolutional stem → four two-convolution residual blocks (32, 64, 128, 256 filters) → Global Average Pooling → Dropout → Softmax | Deeper spatial feature extraction |

The complete layer definitions are in `src/models.py`. Model summaries are saved with each training run.

## Local Training

Run the development experiments locally with the same data, augmentation target, image size, and stopping rules:

```bash
python -m src.train --data-dir data/cleaned_data --model mlp --epochs 20 --learning-rate 0.0001 --verbose 2
python -m src.train --data-dir data/cleaned_data --model compact_cnn --epochs 20 --verbose 2
python -m src.train --data-dir data/cleaned_data --model deeper_cnn --epochs 20 --verbose 2
```

The default optimizer is Adam with zero weight decay. For an independent decoupled weight-decay experiment, use `--optimizer adamw --weight-decay 0.0001`. Optimizer type and decay are recorded in `run_config.json` and validated when resuming a checkpoint.

For long background runs, `--verbose 2` emits one line per epoch instead of rewriting every step, preventing excessive log output.

To resume after an interruption, load the best checkpoint and set `--initial-epoch` to the number of fully completed epochs. For example, resume Compact CNN after eight complete epochs:

```bash
python -m src.train \
  --data-dir data/cleaned_data \
  --model compact_cnn \
  --epochs 20 \
  --resume-from outputs/runs/compact_cnn/best_model.keras \
  --initial-epoch 8 \
  --verbose 2
```

Resume restores model weights and optimizer state, appends `history.csv`, and preserves the previous best validation-accuracy threshold so a worse resumed epoch cannot replace the existing best checkpoint.

The initial ReLU MLP at learning rate 0.001 developed completely inactive hidden layers. The final MLP therefore uses LeakyReLU 0.1 and learning rate 0.0001; the CNNs retain the default 0.001. This is the only intentional model-specific training-setting difference and is recorded in each run configuration.

Development checkpoint selection, early stopping, and learning-rate scheduling all monitor validation Top-1 accuracy in `max` mode. This aligns model selection with the assignment's recognition-rate objective and avoids premature stopping when validation loss rises because of overconfident wrong predictions.

The default float32 policy was used for the successful Metal backend checks. `--mixed-precision` is available as an optional experiment but is not required and should be enabled only after a separate numerical-stability check. Batch size, learning rate, seed, and other settings are configurable through command-line options.

Each run saves:

- `run_config.json`: complete settings, software versions, and physical devices
- `model_summary.txt`: layer architecture and parameter counts
- `history.csv`, `history.json`, and `training_curves.png`: training history
- `best_model.keras`: highest validation-accuracy checkpoint for development runs
- `final_model.keras`: model at the end of the run

After selecting an architecture, retrain it with the complete training pool, including the development-validation images. Use the best development epoch count as a fixed epoch count:

```bash
python -m src.train \
  --data-dir data/cleaned_data \
  --model deeper_cnn \
  --include-validation \
  --epochs 20 \
  --verbose 2 \
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

All reported values below come from the original, non-augmented 162,797-image test set. The initial MLP baseline used validation-loss selection before the selection rule changed; its best checkpoint was epoch 3 and training stopped after epoch 6. Compact CNN reached its best validation accuracy at epoch 8 and stopped after epoch 11. The development Deeper CNN completed all 20 epochs and reached its best validation accuracy at epoch 20.

| Model | Learning rate | Test loss | Test Top-1 | Test Top-5 |
|---|---:|---:|---:|---:|
| MLP with LeakyReLU 0.1 | 0.0001 | 12.2961 | 0.0964% | 0.3219% |
| Compact CNN | 0.001 | 23.7212 | 1.5842% | 3.5080% |
| Deeper CNN, development split | 0.001 | 18.2040 | 16.4763% | 19.4549% |
| **Deeper CNN, complete training pool** | 0.001 | **12.3341** | **16.5636%** | **27.9796%** |

The final submission model is the Deeper CNN retrained for 20 fixed epochs with the complete training pool. Relative to the development Deeper CNN, its Top-1 accuracy is similar, Top-5 accuracy increases from 19.45% to 27.98%, and test loss decreases from 18.20 to 12.33. No validation metrics are reported for this full-pool run because the validation images were returned to training.

The final run reached 98.51% training accuracy, compared with 16.56% test Top-1 accuracy. This gap shows that substantial overfitting remains despite augmentation. Limited originals per class, similar Chinese character shapes, and deterministic repeated augmentation variants are important limitations. More diverse real handwriting and less repetitive augmentation are recommended for future improvement.

The final metrics are stored in `outputs/runs/deeper_cnn_full40/test_evaluation/metrics.json`. A final-model single-image check on the original test image `5/他_41.png` correctly returned `他` with probability 0.5976 and Top-5 candidates `他; 她; 伳; 怹; 崺`. This verifies the end-to-end prediction pipeline and Chinese-character mapping; individual examples are not used to calculate overall accuracy.

Smoke-test results from a reduced or synthetic dataset are debugging evidence only and are not assignment test accuracy.

## Validation-Only Optimization Experiments

Candidate selection uses validation accuracy at equal epoch budgets; the test set is not used for tuning. The epoch-5 comparison is:

| Experiment | Validation Top-1 | Validation Top-5 | Validation loss |
|---|---:|---:|---:|
| Original Deeper CNN, fixed standard augmentation | 11.2438% | 15.1665% | 21.0903 |
| A1: per-epoch standard augmentation | 0.0210% | 0.0938% | 72.2994 |
| A2: per-epoch mild augmentation | 0.0364% | 0.1646% | 57.5262 |
| C1: fixed standard augmentation, learning rate 0.0005 | 2.9066% | 6.3930% | 20.5012 |

A1 and A2 were rejected and were not evaluated on the test set. A1 temporarily reached 3.7447% validation Top-1 at epoch 4 but collapsed at epoch 5. A2 peaked at 1.0046% at epoch 2 and also remained unstable. Neither produced a promising trajectory against the original epoch-5 baseline, so the augmentation direction was stopped.

C1 was paused safely after epoch 3 and later resumed from its checkpoint. Its final validation Top-1 remained below the original baseline, so it was not extended. C2 is an independent AdamW weight-decay screen with the original learning rate. No optimizer candidate is evaluated on the test set unless it is selected by validation. The existing final model and results remain unchanged.

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
