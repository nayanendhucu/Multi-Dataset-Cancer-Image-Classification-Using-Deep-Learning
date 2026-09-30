# Multi-Dataset Cancer Image Classification Using Deep Learning

## Overview

This project explores the application of deep learning and computer vision to cancer-related medical image classification.

The project brings together multiple cancer-related datasets obtained from Kaggle and organizes them into separate categories for experimentation. The notebook includes dataset preparation, directory organization, image preprocessing, augmentation, custom Convolutional Neural Network (CNN) development, model training, validation analysis, model visualization, model saving, and image-based inference.

The primary supervised learning pipeline implemented in the notebook trains a CNN on **brain tumor images** for binary classification.

The project also includes additional cancer-related image datasets and an experimental image-inference section using skin images, demonstrating how the trained model can be applied to uploaded medical images.

> **Important:** This is an educational deep-learning project and is not a medical diagnostic system. The trained model was developed and evaluated on the dataset used in the notebook and should not be interpreted as clinically validated.

---

# Project Objective

The main objective of this project is to explore how Convolutional Neural Networks can be used for medical image classification and to build an end-to-end image classification workflow.

The project focuses on:

* Working with multiple cancer-related datasets
* Downloading and organizing datasets from Kaggle
* Managing image datasets using directory structures
* Preparing images for deep learning
* Applying image normalization and augmentation
* Building a custom CNN architecture
* Performing binary image classification
* Training and validating the model
* Monitoring accuracy and loss
* Saving the trained model
* Loading new images for prediction
* Experimenting with medical image inference

---

# Datasets

The notebook works with five dataset categories/directories:

```text
datasets/
├── brain/
├── lung/
├── breast/
├── skin/
└── kidney/
```

The original notebook downloads five ZIP archives and extracts them into these directories.

The mapping used in the notebook is:

| Archive           | Notebook Directory | Purpose                                                             |
| ----------------- | ------------------ | ------------------------------------------------------------------- |
| `archive (1).zip` | `datasets/brain`   | Brain tumor images                                                  |
| `archive (2).zip` | `datasets/lung`    | Lung-related cancer images                                          |
| `archive (3).zip` | `datasets/skin`    | Skin-related dataset                                                |
| `archive (4).zip` | `datasets/kidney`  | Dataset whose extracted contents are identified as Skin Cancer ISIC |
| `archive (5).zip` | `datasets/breast`  | Breast-related dataset                                              |

The notebook does not provide the original Kaggle dataset URLs for each archive, so the exact source URL for every archive cannot be safely reconstructed from the notebook alone.

---

# Kaggle Dataset Sources

The project uses datasets obtained from Kaggle.

Some of the cancer image collections represented in this project correspond to publicly available Kaggle medical-image datasets.

### Multi Cancer Dataset

Kaggle's Multi Cancer Dataset contains multiple cancer categories, including:

* Brain Cancer
* Breast Cancer
* Kidney Cancer
* Lung and Colon Cancer
* Other cancer categories

The dataset contains JPEG medical images organized into cancer-specific classes.

[Kaggle – Multi Cancer Dataset](https://www.kaggle.com/datasets/obulisainaren/multicancer-dataset?utm_source=chatgpt.com)

### Lung and Colon Cancer Histopathological Images

The LC25000 dataset contains 25,000 histopathological images across five classes, including lung benign tissue, lung adenocarcinoma, lung squamous cell carcinoma, colon adenocarcinoma, and colon benign tissue.

[Kaggle – Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images?utm_source=chatgpt.com)

### Skin Cancer / ISIC

The notebook's extracted `datasets/kidney` directory contains files under:

```text
Skin cancer ISIC The International Skin Imaging Collaboration
```

The ISIC datasets contain dermoscopic/skin-lesion images used for skin-cancer classification research. Kaggle's ISIC datasets include benign and malignant lesion classifications.

[Kaggle – ISIC Skin Cancer Dataset](https://www.kaggle.com/c/isic-2020/data?utm_source=chatgpt.com)

> The notebook's folder naming does not reliably establish the exact Kaggle URL used for every downloaded archive. The URLs above are provided only where the dataset identity can be established from the available evidence.

---

# Technology Stack

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Google Colab
* Convolutional Neural Networks
* ImageDataGenerator
* Kaggle datasets

---

# Project Workflow

```text
Multiple Kaggle Cancer Datasets
              |
              v
       Dataset Download
              |
              v
       Dataset Organization
              |
              v
      Directory Preparation
              |
              v
      Image Preprocessing
              |
              v
     Image Augmentation
              |
              v
       CNN Architecture
              |
              v
     Binary Classification
              |
              v
       Model Training
              |
              v
 Validation & Performance Analysis
              |
              v
        Model Saving
              |
              v
      New Image Inference
```

---

# Dataset Organization

The notebook begins by creating separate directories:

```python
!mkdir datasets/brain
!mkdir datasets/lung
!mkdir datasets/breast
!mkdir datasets/skin
!mkdir datasets/kidney
```

The downloaded ZIP archives are then extracted into their respective directories.

```python
!unzip "archive (1).zip" -d datasets/brain
!unzip "archive (2).zip" -d datasets/lung
!unzip "archive (3).zip" -d datasets/skin
!unzip "archive (4).zip" -d datasets/kidney
!unzip "archive (5).zip" -d datasets/breast
```

This creates a centralized structure for managing the different medical image datasets.

---

# Brain Tumor Dataset Preparation

The primary training dataset is the brain tumor dataset.

Initially, the extracted dataset contains:

```text
brain_tumor_dataset/
├── yes/
└── no/
```

The notebook reorganizes the images into:

```text
datasets/brain/
├── yes/
└── no/
```

The original nested directory is then removed.

This directory structure is compatible with TensorFlow's `flow_from_directory()` functionality.

---

# Image Dataset Loading

The notebook uses Keras' `ImageDataGenerator` to load and preprocess the images.

```python
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    horizontal_flip=True,
    zoom_range=0.1
)
```

Images are:

* Rescaled to the 0–1 range
* Rotated during augmentation
* Horizontally flipped
* Randomly zoomed
* Divided into training and validation subsets

---

# Training and Validation Split

A validation split of:

```text
20%
```

is used.

The notebook reports:

```text
Training images: 203
Validation images: 50
Classes: 2
```

The two classes are represented by:

```text
yes
no
```

corresponding to the brain tumor dataset's class structure.

---

# Image Dimensions

The CNN receives images resized to:

```text
224 × 224 × 3
```

The three channels correspond to RGB image input.

The batch size is:

```text
32
```

---

# Convolutional Neural Network

A custom CNN is implemented using Keras' Sequential API.

The architecture is:

```text
Input
224 × 224 × 3
       |
       v
Conv2D
32 Filters
       |
       v
MaxPooling2D
       |
       v
Conv2D
64 Filters
       |
       v
MaxPooling2D
       |
       v
Flatten
       |
       v
Dense
128 Neurons
       |
       v
Dropout
50%
       |
       v
Dense
1 Neuron
Sigmoid
```

### Main components

* Two convolutional layers
* Two max-pooling layers
* Flatten layer
* Fully connected dense layer
* Dropout regularization
* Sigmoid output layer

The final sigmoid layer is used for binary classification.

---

# Model Compilation

The model is compiled using:

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### Configuration

| Component           | Configuration       |
| ------------------- | ------------------- |
| Optimizer           | Adam                |
| Loss                | Binary Crossentropy |
| Metric              | Accuracy            |
| Output activation   | Sigmoid             |
| Classification type | Binary              |

---

# Model Training

The CNN is trained for:

```text
10 epochs
```

using the training and validation generators.

```python
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)
```

The training process records:

* Training accuracy
* Validation accuracy
* Training loss
* Validation loss

These metrics are stored in the training history for later visualization.

---

# Training Performance Analysis

The notebook visualizes the model's learning process using training and validation curves.

Two performance plots are produced:

### Accuracy

```text
Training Accuracy
Validation Accuracy
```

### Loss

```text
Training Loss
Validation Loss
```

These plots allow the training process to be inspected across epochs and provide an indication of how the model performs on both training and validation images.

---

# Model Saving

After training, the CNN is saved as:

```text
brain_tumor_model.h5
```

The saved model allows the trained network to be reused without retraining it from the beginning.

---

# Brain Tumor Image Inference

The notebook implements an image-upload prediction workflow.

A new brain image is uploaded:

```python
files.upload()
```

The image is then:

1. Loaded
2. Resized to 224 × 224
3. Converted into an array
4. Normalized
5. Expanded to the required model input shape
6. Passed through the CNN
7. Classified using a 0.5 prediction threshold

Example preprocessing:

```python
img = image.load_img(
    img_path,
    target_size=(224,224)
)

img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
```

The notebook then produces one of two outputs:

```text
Tumor Detected
```

or:

```text
No Tumor
```

The recorded notebook run produced:

```text
Prediction: Tumor Detected
```

for the uploaded test image.

---

# Skin Image Inference Experiment

The notebook also contains a separate inference section using skin images.

Several images are placed into:

```text
test_skin/
```

including:

```text
SkinCancerImage.jpg
SkinCancer2.jpg
skinCancer3.jpg
skincancer4.jpg
```

The images are resized and passed through the same trained model.

The notebook then displays a prediction based on a 0.5 threshold:

```python
if prediction[0][0] > 0.5:
    print("Skin Cancer Detected")
else:
    print("Benign Skin")
```

This section demonstrates the mechanics of applying a trained image-classification model to uploaded images.

However, because the primary CNN was trained using the brain tumor dataset, these skin-image predictions should **not** be interpreted as a validated skin-cancer classifier.

---

# Key Machine Learning Concepts Demonstrated

This project covers a broad range of practical deep-learning concepts.

## Data Handling

* Kaggle dataset acquisition
* ZIP archive handling
* Directory organization
* Image dataset preparation
* Class-based directory structures

## Image Processing

* Image loading
* Image resizing
* Pixel normalization
* Image augmentation
* Batch generation
* Training/validation splitting

## Deep Learning

* Convolutional Neural Networks
* Convolution layers
* Max pooling
* Flattening
* Dense layers
* Dropout regularization
* Sigmoid classification

## Model Development

* Binary classification
* Adam optimizer
* Binary crossentropy
* Model training
* Validation
* Training history
* Model serialization

## Evaluation & Visualization

* Training accuracy
* Validation accuracy
* Training loss
* Validation loss
* Learning-curve visualization
* Image prediction visualization

## Deployment-Oriented Workflow

* Saving a trained model
* Loading external images
* Image preprocessing for inference
* Generating predictions from new images

---

# Project Structure

```text
Multi-Dataset-Cancer-Image-Classification/
│
├── datasets/
│   ├── brain/
│   │   ├── yes/
│   │   └── no/
│   │
│   ├── lung/
│   │
│   ├── breast/
│   │
│   ├── skin/
│   │
│   └── kidney/
│
├── test_images/
│   └── braintumorImage.png
│
├── test_skin/
│   ├── SkinCancerImage.jpg
│   ├── SkinCancer2.jpg
│   ├── skinCancer3.jpg
│   └── skincancer4.jpg
│
├── cancer_classification.ipynb
├── brain_tumor_model.h5
└── README.md
```

Large datasets and trained model files may be excluded from GitHub depending on repository size and dataset licensing restrictions.

---

# Project Highlights

This project goes beyond a basic image-classification example by combining several stages of a practical computer-vision workflow:

* Multiple Kaggle dataset sources
* Multi-dataset directory management
* Medical image preprocessing
* Image augmentation
* Custom CNN architecture
* Binary classification
* Training and validation monitoring
* Learning-curve visualization
* Model serialization
* Uploaded-image prediction
* Separate experimental skin-image inference
* GPU-based development using Google Colab

---

# Limitations

This project is intended for educational and portfolio purposes.

The primary CNN was trained using the brain tumor dataset and should therefore be interpreted specifically within the context of that training data.

The notebook does not demonstrate independently trained models for all five dataset directories.

The additional skin-image inference section uses the same trained model and therefore does not establish that the model has learned skin-cancer-specific visual patterns.

The model should not be used to diagnose cancer or make medical decisions.

Real-world medical AI systems require significantly more extensive validation, appropriate clinical datasets, robust evaluation metrics, external validation, clinical expertise, and regulatory consideration.

---

# Future Improvements

The project can be extended into a more comprehensive medical-image classification system.

Potential improvements include:

* Train separate models for each cancer dataset
* Build dedicated brain, lung, breast, skin, and kidney classifiers
* Compare CNN architectures
* Use transfer learning with MobileNetV2, ResNet, EfficientNet, or similar architectures
* Apply systematic hyperparameter tuning
* Address class imbalance
* Add confusion matrices
* Calculate precision, recall, F1-score, and ROC-AUC
* Perform stronger validation
* Add cross-validation where appropriate
* Apply Grad-CAM for model interpretability
* Develop an image-upload web interface
* Deploy models using Streamlit or Flask
* Build a unified multi-cancer image classification application

---

# Conclusion

This project demonstrates an end-to-end deep-learning workflow for cancer-related medical image classification using multiple Kaggle datasets.

The project covers the complete process from dataset acquisition and organization through image preprocessing, augmentation, CNN architecture development, model training, validation analysis, visualization, model saving, and new-image inference.

The primary implemented model performs binary classification on brain tumor images, while the project also contains additional cancer-related datasets and experimental skin-image inference.

The work provides a foundation for extending the system into a more comprehensive multi-dataset medical image classification platform.

---

# Dataset Attribution

The datasets used in this project were obtained from publicly available Kaggle sources. Dataset ownership, attribution requirements, and licenses remain with their respective original authors and dataset providers.

Users of this repository should review and comply with the individual dataset licenses before redistributing any dataset files.

---

# Author

**Nayanendhu CU**
