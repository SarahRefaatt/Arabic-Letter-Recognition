# Arabic Characters Recognition

This project focuses on Arabic characters recognition using various machine learning algorithms such as K-Nearest Neighbors (KNN), Neural Network (NN), and Support Vector Machine (SVM). The dataset used for training and testing consists of Arabic character images.

## Prerequisites

Before running the code, make sure you have the following libraries installed:

- numpy
- pandas
- matplotlib
- scikit-learn
- tensorflow

## Dataset

The dataset used for this project includes the following files:

- "csvTrainImages 13440x1024.csv": CSV file containing training images.
- "csvTrainLabel 13440x1.csv": CSV file containing corresponding labels for the training images.
- "csvTestImages 3360x1024.csv": CSV file containing test images.
- "csvTestLabel 3360x1.csv": CSV file containing corresponding labels for the test images.

## Usage

1. Ensure that the dataset files are located in the correct paths specified in the code.

2. Run the code in a Python environment.

```python
# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.svm import SVC

# Preprocessing
# ...

# KNN
# ...

# Neural Network
# ...

# SVM
# ...

# Results
# ...
```

## Results

The code computes the F1 scores for the evaluated models (SVM, KNN, Neural Network) and determines the best performing model based on the highest F1 score. The results are printed at the end of the code execution.

```
F1 Scores:
SVM: 0.85
KNN: 0.92
Neural Network: 0.93

Best Model: Neural Network
Best Model F1 Score: 0.93
```


## Contact

For any questions or inquiries, please contact me  [sarah.mahmoudd24@gmail.com].