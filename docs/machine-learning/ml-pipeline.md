# 🧠 Machine Learning & Ensemble Pipeline

> **Technical documentation for the tri-model Convolutional Neural Network (CNN) ensemble, dataset preprocessing, cloud weights management, and inference pipeline.**

---

## 📊 Dataset: HAM10000 Overview

The machine learning models are trained on the **HAM10000** (*"Human Against Machine with 10,000 training images"*) dermatoscopy dataset, curated by the International Skin Imaging Collaboration (ISIC).

```
Total Samples: 10,015 dermatoscopic images
Image Modality: Dermoscopy (RGB)
Target Classes: 7 distinct dermatological conditions
```

### Class Definitions & Disease Taxonomy

| Class Code | Full Medical Terminology | Clinical Description | Risk Category |
| :--- | :--- | :--- | :--- |
| **`mel`** | **Melanoma** | Malignant neoplasm of melanocytes; highly aggressive skin cancer. | 🚨 High Risk (Malignant) |
| **`bcc`** | **Basal Cell Carcinoma** | Common non-melanoma malignant skin cancer arising from basal cells. | ⚠️ Moderate Risk (Malignant) |
| **`akiec`** | **Actinic Keratoses / Intraepithelial Carcinoma** | Pre-cancerous or early non-invasive squamous cell carcinoma. | ⚠️ Pre-Cancerous |
| **`bkl`** | **Benign Keratosis-like Lesions** | Solar lentigines, seborrheic keratoses, and lichen-planus like keratoses. | 🟢 Benign |
| **`nv`** | **Melanocytic Nevi** | Common benign skin moles composed of melanocytes. | 🟢 Benign |
| **`df`** | **Dermatofibroma** | Benign fibrous skin nodule typically occurring on extremities. | 🟢 Benign |
| **`vasc`** | **Vascular Lesions** | Angiomas, cherry angiomas, and pyogenic granulomas. | 🟢 Benign |

---

## 🔬 Model Architectures & Transfer Learning

To maximize diagnostic accuracy across diverse lesion textures and pigmentation, the platform implements **heterogeneous ensemble learning** across three foundational deep learning architectures:

```mermaid
graph TD
    InputImage["Input Lesion (RGB, 224x224x3)"]

    subgraph Preprocessing ["Data Preprocessing Pipeline"]
        CV2Read["OpenCV BGR to RGB Conversion"]
        Resize["Bilinear Resize (224x224)"]
        Normalize["EfficientNet Preprocessing [-1, 1] Normalization"]
        CV2Read --> Resize --> Normalize
    end

    subgraph BaseModels ["Base Neural Architectures (Transfer Learning)"]
        M1["EfficientNetB3 (Compound Scaling)"]
        M2["ResNet101 (Deep Residuals)"]
        M3["DenseNet121 (Dense Feature Reuse)"]
    end

    subgraph Aggregation ["Ensemble Aggregation"]
        SoftVote["Soft-Voting Averaging: P_ensemble = 1/N sum(P_i)"]
        MaxConf["Confidence Score Extraction"]
        LabelMap["IDX2CLASS & Clinical Metadata Mapping"]
    end

    InputImage --> Preprocessing
    Normalize --> M1
    Normalize --> M2
    Normalize --> M3

    M1 -->|Probability Vector (1x7)| SoftVote
    M2 -->|Probability Vector (1x7)| SoftVote
    M3 -->|Probability Vector (1x7)| SoftVote

    SoftVote --> MaxConf --> LabelMap --> Output["Final Diagnostic Output JSON"]
```

---

### 1. EfficientNetB3
* **File**: [`backend/app/ml/architectures/efficientnet.py`](file:///d:/repos/skin_disease_detection/backend/app/ml/architectures/efficientnet.py)
* **Base**: Pre-trained `EfficientNetB3` (ImageNet weights, top excluded).
* **Head**: `GlobalAveragePooling2D` &rarr; `BatchNormalization` &rarr; `Dense(512, activation='relu')` &rarr; `Dropout(0.4)` &rarr; `Dense(7, activation='softmax')`.

### 2. ResNet101
* **File**: [`backend/app/ml/architectures/resnet.py`](file:///d:/repos/skin_disease_detection/backend/app/ml/architectures/resnet.py)
* **Base**: Pre-trained `ResNet101` (ImageNet weights).
* **Head**: `GlobalAveragePooling2D` &rarr; `Dense(512, activation='relu')` &rarr; `Dropout(0.5)` &rarr; `Dense(7, activation='softmax')`.

### 3. DenseNet121
* **File**: [`backend/app/ml/architectures/densenet.py`](file:///d:/repos/skin_disease_detection/backend/app/ml/architectures/densenet.py)
* **Base**: Pre-trained `DenseNet121` (ImageNet weights).
* **Head**: `GlobalAveragePooling2D` &rarr; `BatchNormalization` &rarr; `Dense(512, activation='relu')` &rarr; `Dropout(0.3)` &rarr; `Dense(7, activation='softmax')`.

---

## ⚙️ Data Preprocessing & Augmentation Pipeline

### Online Preprocessing (Inference)
* **File**: [`backend/app/services/prediction_service.py`](file:///d:/repos/skin_disease_detection/backend/app/services/prediction_service.py#L60-L70)
* Step 1: Input image decoded via `cv2.imread()`.
* Step 2: Converted from BGR to RGB via `cv2.cvtColor()`.
* Step 3: Resized to `TARGET_IMAGE_SIZE = (224, 224)`.
* Step 4: Normalized via `tf.keras.applications.efficientnet.preprocess_input`.
* Step 5: Expanded to 4D batch tensor `(1, 224, 224, 3)`.

---

## 🗳️ Ensemble Inference Strategy

### Soft-Voting (Production Default)
During real-time inference, the prediction service executes soft-voting consensus:

$$\bar{P}_c = \frac{1}{M} \sum_{m=1}^{M} P_{m, c}$$

Where $M = 3$ (number of models), $P_{m, c}$ is the softmax probability output by model $m$ for class $c$, and $\hat{y} = \arg\max_c (\bar{P}_c)$ yields the predicted disease class.

---

## ☁️ Automated Cloud Weights Downloader

* **File**: [`backend/app/ml/cloud_downloader.py`](file:///d:/repos/skin_disease_detection/backend/app/ml/cloud_downloader.py)
* Automatically fetches missing `.h5` files into `trained_models/` at startup or first inference request:
  * `densenet121.h5` (~49.2 MB)
  * `efficientnetb3.h5` (~90.2 MB)
  * `resnet101.h5` (~256.2 MB)

---

## 🔗 Cross References
* [Model Training Pipeline (`model-training.md`)](./model-training.md)
* [System Architecture (`../architecture/system-architecture.md`)](../architecture/system-architecture.md)
* [REST API Reference (`../api-reference/endpoints.md`)](../api-reference/endpoints.md)
