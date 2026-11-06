# Waste Classification for Recycling

A deep learning web application that classifies a photo of a waste item into one of 13 categories and shows recycling guidance. The classifier is an image model trained with **Google Teachable Machine**; it is the only source of predictions in the whole application.

| | |
|---|---|
| **Program** | B.Tech Deep Learning, Semester V, Section B |
| **Group ID** | BT_DL_09 |
| **Project ID** | DL-009 |
| **Mentor** | Sachin Raghav |
| **University** | Invertis University |

Team members
1. RACHIT CHANDRA - BCS2024127
2. ABHISHEK DIXIT - BCS2024128
3. KAMAL YADAV - BCS2024129
4. DHRUV SHEKHAR - BCS2024130
5. YASH PRATAP SINGH - BCS2024135
6. VANSHIKA GUPTA - BCS2024136
7. PRAKHAR SINGH - BCS2024137
8. LOVEPREET KAUR - BCS2024138
9. RAMANUJ SHANKHDHAR - BCS2024139
10. LAKSHYA BAJPAIE - BCS2024141
11. KARAMJEET KAUR - BCS2024143
12. MOHD ARSHAD - BCS2024146
13. AYAN HUSSAIN - BCS2024147
14. ESHA ARYA - BCS2024148  

---

## 1. What the application does

1. The user uploads an image or takes a photo with the camera (Streamlit web app).
2. The image is validated (type, size, readable) and preprocessed.
3. The exported Teachable Machine model predicts the class.
4. The app shows the predicted class, the confidence, the top-3 predictions, a warning when the confidence is below 60%, and general recycling guidance.

```
Upload / camera  ->  validation + preprocessing  ->  Teachable Machine model  ->  class + confidence  ->  recycling guidance
```

There is **no fallback model**. If the model files are missing or cannot be loaded, the app shows a clear error and makes no prediction.

### Classes (from `models/teachable_machine/labels.txt`)

`battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass, e-waste`

---

## 2. Model

- Trained in the browser with Google Teachable Machine (image project, standard image model, default settings) and exported with **Export Model > TensorFlow > Keras**.
- Files used by the app: `models/teachable_machine/keras_model.h5` and `models/teachable_machine/labels.txt`.
- The layer names inside the exported file show a MobileNetV2-style convolutional backbone, followed by `GlobalAveragePooling2D` and two `Dense` layers.
- Input: 224 x 224 RGB. Preprocessing in `src/predict.py`: centre-crop to a square, resize to 224 x 224 (LANCZOS), scale pixels to the range [-1, 1].
- Teachable Machine exports an older `.h5` format that Keras 3 cannot load directly. `src/tm_model.py` loads it from a temporary patched copy of the model configuration. The original file is never modified.

### Training data

| Class group | Images used in Teachable Machine | Source |
|---|---|---|
| 12 classes (battery ... white-glass) | 250 per class (3,000 total), randomly sampled from the **training split** only (`src/make_tm_dataset.py`, seed 42) | Kaggle "Garbage Classification" by mostafaabla: https://www.kaggle.com/datasets/mostafaabla/garbage-classification (15,515 images, 12 classes) |
| e-waste | Images from a separate e-waste folder (369 images) | **Source: (add the dataset name / link here)** |

---

## 3. Evaluation

### Protocol

- The Kaggle dataset was split once into train / validation / test (70 / 15 / 15, stratified, seed 42; `src/prepare_data.py`).
- Only training-split images were uploaded to Teachable Machine. The **2,328 test images were never shown to the model**.
- All e-waste dataset images were used for training, so e-waste has **no dataset-style test set**. It was tested only on **39 photos taken by the team with a phone**: 20 on a dark cardboard surface (intended as held-out photos) and 19 on other surfaces (taken after the model was trained and never uploaded to Teachable Machine).
- Evaluation uses the same preprocessing and model-loading code as the app (`src/evaluate.py`).

### Results (final model, 2,367 test images)

| Test set | Images | Accuracy |
|---|---|---|
| Dataset test images (12 classes) | 2,328 | **86.38%** |
| Own phone photos of e-waste, dark cardboard | 20 | 25.00% (5 of 20) |
| Own phone photos of e-waste, other surfaces | 19 | 15.79% (3 of 19) |
| All own phone photos of e-waste | 39 | **20.51%** (8 of 39) |
| All test images together | 2,367 | 85.30% |

| Metric (all 2,367 images) | Value |
|---|---|
| Top-3 accuracy | 95.48% |
| Macro precision / recall / F1 | 0.7813 / 0.7589 / 0.7577 |
| Weighted precision / recall / F1 | 0.8601 / 0.8530 / 0.8497 |
| Predictions with confidence >= 60% | 96.9% |
| Accuracy when confidence >= 60% | 87.10% |
| Accuracy when confidence < 60% | 28.77% |

The 12-class dataset accuracy counts a prediction of "e-waste" as an error, because e-waste is one of the 13 possible answers.

### Per-class results

| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| battery | 0.8671 | 0.8732 | 0.8702 | 142 |
| biological | 0.9333 | 0.9459 | 0.9396 | 148 |
| brown-glass | 0.8333 | 0.7143 | 0.7692 | 91 |
| cardboard | 0.7349 | 0.9104 | 0.8133 | 134 |
| clothes | 0.9806 | 0.9474 | 0.9637 | 799 |
| green-glass | 0.9744 | 0.4043 | 0.5714 | 94 |
| metal | 0.7245 | 0.6174 | 0.6667 | 115 |
| paper | 0.8442 | 0.8228 | 0.8333 | 158 |
| plastic | 0.6828 | 0.7615 | 0.7200 | 130 |
| shoes | 0.8379 | 0.9226 | 0.8782 | 297 |
| trash | 0.8407 | 0.9135 | 0.8756 | 104 |
| white-glass | 0.6275 | 0.8276 | 0.7138 | 116 |
| e-waste | 0.2759 | 0.2051 | 0.2353 | 39 |

Most frequent confusions (true -> predicted): clothes -> shoes (21), green-glass -> plastic (18), metal -> white-glass (18), plastic -> white-glass (15), e-waste -> cardboard (14), paper -> cardboard (13), brown-glass -> shoes (12), green-glass -> white-glass (12).

Plots: `outputs/plots/confusion_matrix.png`, `outputs/plots/confusion_matrix_normalized.png`, `outputs/plots/f1_per_class.png`, `outputs/plots/class_distribution.png`. Reports: `outputs/reports/` (`metrics.json`, `classification_report.txt`, `confusion_matrix.csv`, `predictions.csv`).

### Additional checks

- **Preprocessing comparison** (`src/probe_preprocess.py`): five resize / normalisation variants gave 83.3% to 84.7% accuracy on 300 validation images. The difference is within noise, so the original preprocessing was kept.
- **Web preview comparison:** on 4 test images, the Teachable Machine web preview and this Python pipeline gave the same top class for 2 images and a different top class for 2 images (both borderline cases). No tested preprocessing variant reproduced the web result.

---

## 4. Known limitations

- **Poor e-waste recognition on phone photos.** Only 8 of 39 own e-waste photos were classified correctly (5 of 20 on dark cardboard, 3 of 19 on other surfaces). 14 of the 39 were predicted as `cardboard`, often with very high confidence, including 5 of the 19 photos taken on other surfaces. The surface may play a part, but the model also seems to struggle with real phone photos in general.
- **Weak classes.** `green-glass` (recall 40.4%), `metal`, `plastic` and `white-glass` have the lowest scores. Glass types and shiny materials are often mixed up.
- **Over-confidence.** 96.9% of predictions have a confidence of at least 60%, but only 87.1% of those are correct, so the 60% warning catches few mistakes.
- **E-waste test set is small.** 39 photos from two sessions. The 20.5% result is a rough indicator, not a precise estimate.
- **Closed set.** The model knows only its 13 classes. Any other object is forced into one of them.
- **Dataset.** The Kaggle images come mostly from the web, so real-world photos can be harder. Near-duplicate images between the dataset splits could make the dataset test accuracy slightly optimistic; this was not checked.
- **Recycling guidance is general.** Local waste rules differ by city and municipality.

---

## 5. Project structure

```
waste-classification/
|-- app/streamlit_app.py          Streamlit web application
|-- src/
|   |-- config.py                 Paths and settings
|   |-- tm_model.py               Loads the Teachable Machine model (no fallback)
|   |-- predict.py                Validation, preprocessing, prediction
|   |-- recycling_info.py         Recycling guidance per class
|   |-- evaluate.py               Evaluation on the held-out test images
|   |-- prepare_data.py           70/15/15 split of the Kaggle dataset
|   |-- explore_data.py           Class counts and corrupt-image check
|   |-- make_tm_dataset.py        Builds the image subset uploaded to Teachable Machine
|   |-- probe_preprocess.py       Preprocessing comparison experiment
|   |-- check_tm_model.py         Checks that the model loads
|   |-- inspect_tm_h5.py          Prints the structure of the exported .h5 file
|   `-- check_env.py              Checks the TensorFlow installation
|-- tests/                        pytest tests (12 tests)
|-- models/teachable_machine/     keras_model.h5, labels.txt
|-- outputs/plots, outputs/reports
|-- docs/                         Screenshots and project documents
|-- requirements.txt              Tested package versions
`-- requirements-lock.txt         Exact versions of the development environment
```

---

## 6. Setup and run

**Requirements:** Python **3.12** (tested on Windows 11). TensorFlow 2.21.0 does not install on Python 3.14.

```powershell
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once.

Check that the model loads (expects 13 classes):

```powershell
python -m src.check_tm_model
```

Start the web app (open http://localhost:8501):

```powershell
streamlit run app/streamlit_app.py
```

Classify a single image from the command line:

```powershell
python -m src.predict path\to\image.jpg
```

Run the tests:

```powershell
python -m pytest -v
```

The app works without the dataset. The dataset is needed only to recreate the split and the evaluation.

### Recreating the evaluation (optional)

1. Download the Kaggle dataset and extract it to `data/raw/garbage_classification/` (12 class folders).
2. `python -m src.explore_data`
3. `python -m src.prepare_data`
4. `python -m src.evaluate`

The e-waste test photos (`data/extra_test/e-waste/`) are not included in this repository. Without them, `evaluate.py` reports the 12 dataset classes only and prints which classes were not evaluated.

---

## 7. Screenshots

| Home page | Prediction | Error handling |
|---|---|---|
| ![Home](docs/screenshots/app/home.png) | ![Prediction](docs/screenshots/app/prediction.png) | ![Error](docs/screenshots/app/error.png) |

---

## 8. Troubleshooting

| Message | Meaning / fix |
|---|---|
| `Teachable Machine file not found` | `keras_model.h5` or `labels.txt` is missing from `models/teachable_machine/` |
| `Could not load the Teachable Machine model` | The model file is damaged or the TensorFlow / Keras version is different. Use the versions in `requirements.txt` |
| `No matching distribution found for tensorflow` | Use Python 3.12, not 3.14 |

---

## 9. Credits

- Image model: Google Teachable Machine (https://teachablemachine.withgoogle.com).
- Dataset: "Garbage Classification" by mostafaabla, Kaggle (12 classes).
- Built with TensorFlow / Keras, Streamlit, scikit-learn, Pillow, pandas, matplotlib and seaborn.
