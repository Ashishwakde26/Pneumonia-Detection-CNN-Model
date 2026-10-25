# Pneumonia Detection Using a Convolutional Neural Network (CNN)

A deep learning project that classifies chest X-ray images into **NORMAL** and **PNEUMONIA** using a Convolutional Neural Network (CNN) built with **TensorFlow/Keras**.

> **Disclaimer:** This project is for learning and demonstration only. It is not clinically validated and must not be used to diagnose pneumonia or make medical decisions.

## Project overview

This repository demonstrates an end-to-end image-classification workflow: loading chest X-rays, preprocessing images, splitting datasets, applying data augmentation, constructing and training a CNN, evaluating its performance, and visualizing predictions.

### What the CNN learns

Convolutional filters learn image patterns and produce feature maps. Batch normalization helps stabilize training, max pooling reduces spatial dimensions, and global average pooling summarizes the final feature maps. Dense layers then generate a probability for binary classification.

## Model architecture

Input images are grayscale with shape **150 × 150 × 1**.

| Stage | Layer | Output / purpose |
|---|---|---|
| Input | Grayscale X-ray | 150 × 150 × 1 |
| Block 1 | Conv2D(32, 3×3, same, ReLU) → BatchNormalization → MaxPooling2D | 32 learned feature maps, downsampled |
| Block 2 | Conv2D(64, 3×3, same, ReLU) → BatchNormalization → MaxPooling2D | 64 feature maps |
| Block 3 | Conv2D(128, 3×3, same, ReLU) → BatchNormalization → MaxPooling2D | 128 feature maps |
| Block 4 | Conv2D(256, 3×3, same, ReLU) → BatchNormalization → MaxPooling2D | 256 feature maps |
| Pooling | GlobalAveragePooling2D | 256 values per image |
| Classifier | Dense(128, ReLU) → Dropout(0.5) | 128 neurons; dropout during training |
| Output | Dense(1, sigmoid) | Probability between 0 and 1 |

The model uses the **Adam** optimizer, **binary cross-entropy** loss, and **accuracy** as a training metric.

<img width="1607" height="849" alt="image" src="https://github.com/user-attachments/assets/27c7f762-4a40-4965-82db-f38359e60d4c" />


## Dataset and preprocessing

The dataset contains two labeled classes: `NORMAL` and `PNEUMONIA`. Images are read as grayscale, resized to **150 × 150**, converted into NumPy arrays, and reshaped into `(samples, 150, 150, 1)` for CNN input.

Ensure that image intensities are scaled consistently across training, validation, and testing (for example, divide by 255 if your pipeline has not already normalized them).

The project uses a stratified validation split:

```python
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)
```

**Important:** Keep the test set separate from training and validation. If multiple X-rays belong to the same patient, split by patient to avoid leakage.

## Data augmentation

Training images are augmented on the fly using Keras `ImageDataGenerator`:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False
)
```

Augmentation generates transformed batches during training rather than permanently adding files to the dataset. For medical images, augmentation choices should be checked for clinical plausibility.

## Training

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=18,
    validation_data=(x_val, y_val),
    callbacks=[learning_rate_reduction]
)
```

`learning_rate_reduction` is a `ReduceLROnPlateau` callback that lowers the learning rate when validation accuracy stops improving.

<img width="1842" height="833" alt="image" src="https://github.com/user-attachments/assets/070dca00-fb60-4ce1-801e-f77bd24df577" />


## Evaluation and visualization

Plot training and validation accuracy to understand learning progress:

```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()
```

<img width="856" height="546" alt="image" src="https://github.com/user-attachments/assets/8c847049-5359-4986-8a63-1dbac9ed0889" />

Generate predictions on the **held-out test set**:

```python
probabilities = model.predict(x_test).ravel()
predictions = (probabilities > 0.5).astype('int32')

for i in range(min(15, len(predictions))):
    print(f'Actual: {y_test[i]}, Predicted: {predictions[i]}, Probability: {probabilities[i]:.2f}')
```

**Label mapping:** Confirm whether `1` represents `NORMAL` or `PNEUMONIA` in your dataset before interpreting sigmoid probabilities.

For a medical-imaging project, also report a confusion matrix, precision, recall/sensitivity, F1 score, and ideally ROC-AUC on the untouched test set. Overall accuracy alone can be misleading with imbalanced classes.

## Getting started

### Prerequisites

- Python 3.10+ with a compatible TensorFlow version
- Jupyter Notebook or JupyterLab
- TensorFlow/Keras, NumPy, OpenCV, Matplotlib, scikit-learn

### Installation

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

pip install tensorflow numpy opencv-python matplotlib scikit-learn jupyter
jupyter notebook
```

Open your project's `.ipynb` notebook, update dataset paths to match your local directory, and run the cells in order.

### Suggested dataset layout

```text
dataset/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Adapt the layout to the dataset and paths used in your notebook. Do not commit large datasets, virtual environments, or trained model files unless intentional.

## Technologies used

**Python · TensorFlow · Keras · NumPy · OpenCV · Matplotlib · scikit-learn · Jupyter Notebook**

## Video walkthrough

Watch the full project explanation: **[YouTube tutorial](https://youtu.be/M8HKrPzPBNw)**

The walkthrough covers convolution and feature maps, batch normalization, max pooling, global average pooling, dense layers, dropout, training, validation, and predictions.


## Limitations and next steps

This model's performance may vary across hospitals, scanners, patient populations, and image acquisition conditions. Dataset bias, class imbalance, and data leakage must be considered before interpreting results. Future improvements include patient-level splitting, external validation, calibration, error analysis, and explainability visualizations such as Grad-CAM.

## License

Add a `LICENSE` file if you want to specify reuse permissions. Check the dataset's license before redistributing any images.
