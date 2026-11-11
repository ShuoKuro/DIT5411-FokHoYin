# DIT5411 Traditional Chinese Handwriting Recognition

A Python and TensorFlow project for the DIT5411 Machine Learning character recognition assignment. The project trains and compares several multilayer classifiers for **13,065 Traditional Chinese character classes**.

## Assignment Requirements

- Use the full Traditional Chinese Handwriting Dataset with 13,065 classes.
- Use the first 40 images of each class, in natural filename order, as the training pool and all remaining images as the test set.
- Generate at least 200 training samples per class using image augmentation.
- Train several multilayer TensorFlow models with a 13,065-neuron softmax output layer.
- Evaluate every model on the same original, non-augmented test images.
- Document the augmentation method, model architectures, measured test accuracies, and references.

The repository name follows the required `DIT5411-your-name` format. All project documentation and program output are in English, except for Chinese class labels and source quotations.

## Dataset

Use the **full 13,065-class, 300×300 dataset**, not the smaller 4,803-class common-character dataset. The official dataset description reports 684,677 images.

Sources:

- [Dataset documentation](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset)
- [Full segmented dataset archive](https://github.com/chenkenanalytic/handwritting_data_all)

A Linux or Colab download workflow is:

```bash
mkdir -p data/source
git clone https://github.com/chenkenanalytic/handwritting_data_all.git data/source/handwritting_data_all
cat data/source/handwritting_data_all/all_data.zip* > data/all_data.zip
unzip -O big5 data/all_data.zip -d data
```

The extracted class directories should be located at `data/cleaned_data`. The `-O big5` option preserves the original Chinese filenames. Adjust the path if the archive extracts to a different location. Raw data and generated archives are excluded from Git.

The dataset is distributed under **CC BY-NC-SA 4.0**. Retain attribution to AI-FREE Team and the original STUST EECS Chinese MNIST source when sharing this work.

## Environment

Python 3.11 is recommended. TensorFlow is not assumed to be preinstalled locally.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the full experiments, use a GPU-enabled Colab runtime. The included `notebooks/DIT5411_Colab.ipynb` contains the complete workflow. Mixed precision is optional and should only be enabled on a compatible GPU.

## Deterministic Data Split

Create the manifests and audit the class counts:

```bash
python -m src.prepare_data \
  --data-dir data/cleaned_data \
  --output-dir outputs/data \
  --verify-images
```

Preparation stops with an error unless there are exactly 13,065 class directories and every class has at least 41 images. Images are ordered with a natural filename sort, so numeric suffixes are ordered numerically rather than lexicographically.

| Manifest | Purpose | Images per class |
|---|---|---:|
| `fit.csv` | Development-model training originals | 36 |
| `validation.csv` | Early stopping and learning-rate scheduling | 4 |
| `train.csv` | Complete assignment training pool | 40 |
| `test.csv` | Final recognition evaluation | All remaining images |

The validation images come only from the first 40 training images. Test images are never used for training, augmentation, early stopping, or learning-rate scheduling. Development runs train on `fit.csv`; the final best architecture can be retrained on `train.csv` with `--include-validation`.

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

## Training

Run the following development experiments with the same data, augmentation target, image size, and stopping rules:

```bash
python -m src.train --data-dir data/cleaned_data --model mlp --epochs 20
python -m src.train --data-dir data/cleaned_data --model compact_cnn --epochs 20
python -m src.train --data-dir data/cleaned_data --model deeper_cnn --epochs 20
```

Add `--mixed-precision` to each command when using a compatible GPU. Batch size, learning rate, seed, and other settings are configurable through command-line options. Each run saves:

- `run_config.json`: complete settings, software versions, and hardware information
- `model_summary.txt`: layer architecture and parameter counts
- `history.csv`, `history.json`, and `training_curves.png`: training history
- `best_model.keras`: best validation-loss checkpoint for development runs
- `final_model.keras`: model at the end of the run

After selecting an architecture, retrain it with all 40 original training images per class. Use the best development epoch count as a fixed epoch count; no early stopping is used when validation images are included in training.

```bash
python -m src.train \
  --data-dir data/cleaned_data \
  --model deeper_cnn \
  --include-validation \
  --epochs 20 \
  --output-dir outputs/runs/deeper_cnn_full40
```

Replace the model name and epoch count with the selected settings. This full-pool run supplies exactly 200 samples per class per epoch and saves `last_model.keras` after every completed epoch.

## Test Evaluation

Evaluate each development model on `test.csv`. These are the remaining original images for every class, with no augmentation.

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

No full-dataset recognition accuracy is claimed yet. The local development environment has neither the full dataset nor a GPU training session. Before submission, run all full-dataset experiments and replace this section with a comparison table containing each model's measured test loss, Top-1 accuracy, Top-5 accuracy, epoch count, and a link to its run configuration. Use only values from the generated `metrics.json` files and identify the highest measured Top-1 model as the best model.

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

- `src/prepare_data.py`: full-dataset audit and deterministic split creation
- `src/data.py`: manifest handling, exact sample expansion, and `tf.data` pipelines
- `src/augment.py`: reproducible affine augmentation and preview generation
- `src/models.py`: MLP, Compact CNN, and Deeper Residual CNN definitions
- `src/train.py`: configurable training, checkpoints, and history export
- `src/evaluate.py`: streaming test evaluation and Top-5 prediction export
- `src/predict.py`: single-image inference
- `notebooks/DIT5411_Colab.ipynb`: GPU-oriented end-to-end workflow

The `data/`, `outputs/`, virtual-environment, and cache directories are ignored. Do not commit the raw dataset, augmented dataset, model binaries, or large prediction files. Record the actual software and hardware details from `run_config.json` with the final results.

## References

1. AI-FREE Team, [Traditional Chinese Handwriting Dataset](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset), 2020.
2. [Full dataset archive and deployment example](https://github.com/chenkenanalytic/handwritting_data_all).
3. [TensorFlow Keras documentation](https://www.tensorflow.org/api_docs/python/tf/keras).
4. Darshita Jain, [Introduction to OpenCV](https://github.com/darshitajain/Introduction-to-OpenCV), augmentation reference supplied with the assignment.
5. [Handwritten Digit Recognition using Keras and TensorFlow](https://github.com/kimanalytics/Handwritten-Digit-Recognition-using-Keras-and-TensorFlow), model-documentation reference supplied with the assignment.
