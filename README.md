# Emergency Vehicle Image Classifier

## Project Overview
This project aims to develop and optimize a Convolutional Neural Network (CNN) for classifying images as either 'emergency vehicles' or 'non-emergency vehicles'. The end goal is to build a robust model that can be integrated into a simple application for real-time classification.

## Table of Contents
1.  [Setup and Dependencies](#setup-and-dependencies)
2.  [Data Description](#data-description)
3.  [Data Loading and Preprocessing](#data-loading-and-preprocessing)
4.  [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5.  [Data Augmentation](#data-augmentation)
6.  [Model Development and Tuning](#model-development-and-tuning)
    *   [Model Comparison Summary](#model-comparison-summary)
    *   [Baseline Model](#baseline-model)
    *   [Regularization (Dropout 0.5)](#regularization-dropout-05)
    *   [Regularization (Dropout 0.3)](#regularization-dropout-03)
    *   [Hyperparameter Tuning (Learning Rate 0.0001)](#hyperparameter-tuning-learning-rate-0001)
    *   [Hyperparameter Tuning (Batch Size 64)](#hyperparameter-tuning-batch-size-64)
    *   [Early Stopping](#early-stopping)
7.  [Final Model Selection](#final-model-selection)
8.  [Model Saving and Prediction](#model-saving-and-prediction)
9.  [Project Structure and Organization](#9-project-structure-and-organization)
    *   [Development of the Prediction Module](#development-of-the-prediction-module)
    *   [Streamlit Application Development](#streamlit-application-development)
    *   [Integration and Testing](#integration-and-testing)
    *   [Deployment Preparation](#deployment-preparation)
    *   [Installation](#installation)
    *   [Application Preview](#application-preview)
    *   [Future Improvements](#future-improvements)
    *   [Conclusion](#conclusion)

---

## Setup and Dependencies
This project uses several Python libraries for data manipulation, deep learning, and visualization. The key libraries and their roles are listed below:

*   **`pandas`**: For data manipulation and analysis, especially for handling CSV files.
*   **`numpy`**: For numerical operations, particularly with arrays and mathematical functions.
*   **`matplotlib.pyplot`**: For creating static, interactive, and animated visualizations.
*   **`seaborn`**: For statistical data visualization, built on Matplotlib.
*   **`os`**: For interacting with the operating system, such as creating directories and managing file paths.
*   **`shutil`**: For high-level file operations, like copying files.
*   **`tensorflow.keras`**: The primary deep learning framework for building, training, and evaluating neural networks.
    *   `image_dataset_from_directory`: Utility for loading image datasets from directory structures.
    *   `ImageDataGenerator`: For real-time data augmentation.
    *   `Sequential`: For building linear stack of layers for the neural network.
    *   `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, `Dropout`, `AveragePooling2D`, `BatchNormalization`: Various layers used in the CNN architecture.
    *   `EarlyStopping`: Callback to stop training when a monitored metric has stopped improving.
    *   `Adam`: An optimization algorithm for training neural networks.
    *   `load_model`: For loading a saved Keras model.
*   **`sklearn.preprocessing.LabelEncoder`, `OneHotEncoder`**: For encoding categorical labels (though not explicitly used for the final model labels in this notebook, typically useful).
*   **`sklearn.metrics`**: For evaluating model performance (accuracy, precision, recall, confusion matrix).

---

## Data Description
The dataset consists of images of vehicles, labeled as either 'emergency' or 'non_emergency'.

*   **`train.zip`**: Contains the following:
    *   **`train.csv`**: A CSV file with two columns: `image_names` (filename of each image) and `emergency_or_not` (a binary label, where `0` represents non-emergency and `1` represents emergency). This file contains labels for 1646 (70%) training images.
    *   **`images` folder**: Contains a total of 2352 images, which are used for both the training and testing sets.
    *   **`test.csv`**: A CSV file with one column: `image_names` (filenames of the 706 (30%) test images without labels).

---

## Data Loading and Preprocessing

1.  **Load Labels**: The `train.csv` file is loaded into a pandas DataFrame to get image names and their corresponding labels. The first few rows of this DataFrame are displayed to show the structure of the data.

2.  **Organize Images into Directories**: To prepare the images for TensorFlow's `image_dataset_from_directory` utility, which expects data in a structured format, two subdirectories (`dataset/emergency` and `dataset/non_emergency`) are created. Images are then iterated through based on the `train_df` DataFrame. Each image from the source `train` folder is copied into the appropriate `dataset/emergency` or `dataset/non_emergency` subdirectory according to its `emergency_or_not` label. This step ensures that the image files are correctly categorized into folders representing their classes.

3.  **Create Image Datasets**: TensorFlow's `image_dataset_from_directory` utility is used to load images directly from the organized `dataset` directory. Labels are automatically inferred from the subdirectory names ('emergency' and 'non_emergency'). A 10% validation split is applied to separate the data into training and testing sets.

    Output:
    ```
    Found 1646 files belonging to 2 classes.
    Using 1482 files for training.
    Found 1646 files belonging to 2 classes.
    Using 164 files for validation.
    ['emergency', 'non_emergency']
    ```

4.  **Extract Data and Labels**: The `tf.data.Dataset` objects created in the previous step are iterated over to extract images and their corresponding labels, converting them into NumPy arrays (`x_train`, `y_train`, `x_test`, `y_test`). This prepares the data for further preprocessing and model training.

---

## Exploratory Data Analysis (EDA)

Basic data exploration includes checking the shapes of the loaded arrays and visualizing a few sample images.

*   **Data Shapes**:
    The shapes of the training and testing image arrays (`x_train`, `x_test`) and their corresponding label arrays (`y_train`, `y_test`) are printed to confirm the dimensions of the dataset.

    Output:
    ```
    (1482, 256, 256, 3)
    (1482,) 
    (164, 256, 256, 3)
    (164,)
    ```

*   **Sample Image Visualization**:
    A few sample images from the training set are displayed using `matplotlib.pyplot` to visually inspect the data and confirm correct loading and labeling.

*   **Normalization**: Pixel values are scaled from their original range of `[0, 255]` to `[0, 1]` by dividing by 255. This normalization step is crucial for deep learning models, as it helps in faster convergence and better performance.

---

## Data Augmentation

`ImageDataGenerator` from Keras is used to create augmented images on-the-fly during training. This technique helps in reducing overfitting and improving model generalization by exposing the model to a wider variety of images, even with a limited dataset. The augmentation applies random transformations such as rotations, shifts, shear, and zoom to the training images.

Parameters used:
*   `rotation_range=20`
*   `width_shift_range=0.1`
*   `height_shift_range=0.1`
*   `horizontal_flip=True`
*   `vertical_flip=False`
*   `shear_range=0.10`
*   `zoom_range=0.10`
*   `validation_split=0.2`

An example of a transformed image is displayed to show the effect of the augmentation.

---

## Model Development and Tuning

An iterative approach was used to develop and tune the CNN model, starting with a baseline and progressively introducing regularization and hyperparameter adjustments. Each step involved training the model for 20 epochs and evaluating its performance using accuracy, precision, recall, and confusion matrices.

### Model Comparison Summary

| Model Variation              | Accuracy (%) | Precision (%) | Recall (%) |
| :--------------------------- | :----------- | :------------ | :--------- |
| Baseline Model               | 82.93        | 88.46         | 67.65      |
| Regularization (Dropout 0.5) | 79.88        | 87.23         | 60.29      |
| Regularization (Dropout 0.3) | 80.49        | 87.50         | 61.76      |
| Hyperparameter (LR 0.0001)   | 77.44        | 86.05         | 54.41      |
| Hyperparameter (BS 64)       | 90.24        | 94.83         | 80.88      |
| Early Stopping               | 79.88        | 79.66         | 69.12      |

### Baseline Model

The baseline CNN consists of three convolutional blocks followed by a flatten layer and two dense layers. It serves as a starting point to assess initial performance.

**Architecture**:
*   Input: (256, 256, 3)
*   Conv2D (32 filters, 3x3, 'relu', 'same')
*   MaxPooling2D (2x2, 'same')
*   Conv2D (64 filters, 3x3, 'relu', 'same')
*   MaxPooling2D (2x2, 'same')
*   Conv2D (128 filters, 3x3, 'relu', 'same')
*   MaxPooling2D (2x2, 'same')
*   Flatten
*   Dense (64 units, 'relu')
*   Dense (1 unit, 'sigmoid') - Output layer for binary classification

**Compilation**: The model is compiled using the Adam optimizer with default learning rate, `binary_crossentropy` as the loss function, and `accuracy` as the evaluation metric.
**Training**: The model is trained for 20 epochs with a batch size of 32, using the augmented training data.

**Results Summary**:
The baseline CNN achieved an accuracy of **82.93%**, with a precision of **88.46%** and a recall of **67.65%**. The model showed good reliability in predicting Emergency cases but frequently missed actual emergencies (lower recall). Loss curves indicated effective learning with minor fluctuations, suggesting a tendency towards overfitting. A confusion matrix is generated and plotted to visualize the model's performance on true positives, true negatives, false positives, and false negatives.

### Regularization (Dropout 0.5)

Dropout was introduced to mitigate overfitting observed in the baseline model. A dropout rate of 0.5 was applied after the first dense layer. The model was re-trained and evaluated following the same procedure as the baseline.

**Architecture Change**:
*   Added a Dropout layer with a rate of 0.5 after the first Dense layer (`Dense(64, activation='relu')`).

**Results Summary**:
With dropout at 0.5, accuracy slightly decreased to **79.88%**, precision remained high at **87.23%**, but recall dropped further to **60.29%**. This strong dropout rate led to underfitting, causing the model to miss more true emergency cases. Loss curves indicated increased training loss and more fluctuating validation loss, suggesting that the model was overly constrained. A confusion matrix is generated and plotted.

### Regularization (Dropout 0.3)

Recognizing that a 0.5 dropout rate was too aggressive, the dropout rate was tuned down to 0.3 to find a better balance between regularization and learning capacity. The model was re-trained and evaluated.

**Architecture Change**:
*   Changed the Dropout layer rate from 0.5 to 0.3.

**Results Summary**:
Reducing the dropout rate to 0.3 resulted in an accuracy of **80.49%**, precision of **87.50%**, and recall of **61.76%**. This configuration showed a slight improvement in recall compared to dropout 0.5, indicating a better balance. The loss curves were more stable, suggesting effective regularization without severe underfitting. A confusion matrix is generated and plotted.

### Hyperparameter Tuning (Learning Rate 0.0001)

The learning rate for the Adam optimizer was reduced to 0.0001 to assess its impact on convergence and stability, especially given the fluctuations in previous validation loss curves. The model was re-trained and evaluated.

**Architecture Change**:
*   The Adam optimizer's learning rate was set to 0.0001 during model compilation.

**Results Summary**:
Lowering the learning rate to 0.0001 led to a decline in performance: accuracy dropped to **77.44%**, precision to **86.05%**, and recall to **54.41%**. The model showed slower and less effective convergence, suggesting it was underfitting and couldn't reach a good minimum within the given epochs. The default learning rate (0.001) was deemed more appropriate. A confusion matrix is generated and plotted.

### Hyperparameter Tuning (Batch Size 64)

The batch size for training was increased from 32 to 64 to investigate its effect on gradient estimation and training stability. The model was re-trained and evaluated.

**Architecture Change**:
*   The batch size used during model fitting was changed to 64.

**Results Summary**:
Increasing the batch size to 64 resulted in a substantial improvement across all metrics: accuracy rose to **90.24%**, precision to **94.83%**, and recall to **80.88%**. This configuration achieved the best balance between sensitivity and reliability. The loss curves became significantly smoother and more stable, demonstrating better generalization and effective learning. This batch size proved to be the most effective optimization. A confusion matrix is generated and plotted.

### Early Stopping

Early stopping was introduced as a callback during training to prevent overfitting by halting training when validation loss stops improving, with a patience of 5 epochs. The model was re-trained and evaluated.

**Architecture Change**:
*   An `EarlyStopping` callback was configured to monitor `val_loss` with a patience of 5 epochs and restore best weights, then passed to the `callbacks` argument during model fitting.

**Results Summary**:
Early stopping unexpectedly led to a decline in model performance: accuracy dropped to **79.88%**, precision to **79.66%**, and recall to **69.12%**. This indicated that the model stopped learning prematurely before fully converging and refining its feature representations. For this dataset and architecture, early stopping proved detrimental, suggesting the model needed the full 20 epochs to stabilize and perform optimally with a batch size of 64. A confusion matrix is generated and plotted.

---

## Final Model Selection

Through the iterative tuning process, the model trained with a **batch size of 64**, a **dropout rate of 0.3**, and the **default Adam learning rate (0.001)** consistently delivered the strongest and most balanced performance. It achieved:

### Model Comparison Summary

| Model Variation              | Accuracy (%) | Precision (%) | Recall (%) |
| :--------------------------- | :----------- | :------------ | :--------- |
| Baseline Model               | 82.93        | 88.46         | 67.65      |
| Regularization (Dropout 0.5) | 79.88        | 87.23         | 60.29      |
| Regularization (Dropout 0.3) | 80.49        | 87.50         | 61.76      |
| Hyperparameter (LR 0.0001)   | 77.44        | 86.05         | 54.41      |
| Hyperparameter (BS 64)       | 90.24        | 94.83         | 80.88      |
| Early Stopping               | 79.88        | 79.66         | 69.12      |

These metrics, supported by stable loss curves and a well-balanced confusion matrix, demonstrate superior generalization and reliable class separation. This configuration is selected as the final model for the emergency versus non-emergency image classification task.

---

## Model Saving and Prediction

### Retraining the Best Model
The best performing model configuration (Dropout 0.3, Batch Size 64, default Adam learning rate) was retrained using the augmented training data to ensure consistency and availability for deployment.

### Saving the Model
The trained model is saved in the Keras format (`.keras`) to a designated 'model' directory for future use and deployment. This allows the model to be loaded and used for predictions without needing to retrain it.

### Loading the Model and Prediction Function
A function named `predict_vehicle` is defined to facilitate predictions on new images. This function first loads the saved Keras model. It then takes an image file path as input, loads the image, resizes it to the expected input dimensions (256x256), converts it into a NumPy array, normalizes its pixel values to the `[0, 1]` range, and expands its dimensions to match the model's input shape. Finally, it uses the loaded model to predict the probability of the image being an emergency vehicle and returns the predicted label ('Emergency Vehicle' or 'Non-Emergency Vehicle') along with the raw probability.

### Testing the Prediction Function
The `predict_vehicle` function was tested with example image paths, one of a known emergency vehicle and another of a known non-emergency vehicle, to demonstrate its functionality and accuracy.

*   **Emergency Vehicle Example**:
    When tested with an image of an emergency vehicle, the function correctly predicted it as an 'Emergency Vehicle' with a high probability (e.g., 0.9993495).

*   **Non-Emergency Vehicle Example**:
    When tested with an image of a non-emergency vehicle, the function correctly predicted it as a 'Non-Emergency Vehicle' with a low probability (e.g., 0.20648684).

---

## 9. Project Structure and Organization

To improve maintainability and prepare the project for deployment, the project was reorganized into a structured directory layout.

## Folder Descriptions

| Folder/File             | Purpose                                                                                                                                              |
| :---------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app/`                  | Contains the Streamlit web application (`app.py`) that provides the user interface for image upload and prediction.                                  |
| `model/`                | Stores the trained CNN model saved in Keras format (`.keras`).                                                                                       |
| `src/`                  | Contains reusable Python modules, including the prediction pipeline (`predict.py`).                                                                    |
| `Notebooks_code/`       | Contains the Jupyter Notebook used for data preprocessing, model development, training, evaluation, and experimentation.                             |
| `train/`                | Contains the training images used to train the CNN model.                                                                                            |
| `test/`                 | Contains the testing images used for model evaluation and prediction testing.                                                                        |
| `train.csv`             | Metadata file containing training image labels.                                                                                                      |
| `test.csv`              | Metadata file containing testing image information.                                                                                                  |
| `requirements.txt`      | Lists all Python dependencies required to run the project.                                                                                           |
| `README.md`             | Provides project documentation, setup instructions, and implementation details.                                                                      |

This structure separates model development, prediction functionality, and deployment components, making the project easier to maintain and extend.

### Development of the Prediction Module

A dedicated prediction module (predict.py) was created within the src directory.

The module is responsible for:

*   Loading the trained model
*   Preprocessing uploaded images
*   Generating prediction probabilities
*   Returning classification labels

The prediction workflow consists of:

*   Loading the saved Keras model.
*   Resizing uploaded images to 256 × 256 pixels.
*   Converting images into NumPy arrays.
*   Normalizing pixel values to the range [0,1].
*   Expanding image dimensions.
*   Generating prediction probabilities.
*   Assigning class labels based on a threshold of 0.5.

### Streamlit Application Development

To make the model accessible through a graphical user interface, a web application was developed using Streamlit.

**Features**
*   Upload JPG, JPEG, or PNG images.
*   Display uploaded images.
*   Perform real-time classification.
*   Display prediction labels.
*   Display confidence scores.

The application acts as a bridge between users and the trained CNN model, allowing predictions without requiring any programming knowledge.

### Integration and Testing

The prediction module and Streamlit application were integrated and tested locally.

The following functionality was verified:

*   Successful image upload.
*   Successful model loading.
*   Accurate prediction generation.
*   Correct display of labels.
*   Correct display of confidence scores.

Testing confirmed that the deployed application produced predictions consistent with the results obtained during model evaluation.

### Deployment Preparation

To prepare the project for deployment and sharing, all dependencies were documented in a `requirements.txt` file.

This ensures that the project can be reproduced and executed in different environments by installing the required packages.

The final system consists of:

*   A trained CNN model.
*   A reusable prediction module.
*   A Streamlit web application.

### Installation

**Clone the Repository**
```bash
git clone <repository-url>
cd Emergency_Vehicles_Classifier
Install Dependencies

pip install -r requirements.txt
Run the Application

streamlit run app/app.py
```

### Application Preview
Home Page
<img width="468" height="275" alt="image" src="https://github.com/user-attachments/assets/f6da4b01-0307-4c6d-a1ec-c1d72ad0aa68" />

Emergency Vehicle Prediction

<img width="917" height="612" alt="image" src="https://github.com/user-attachments/assets/c0a9d9fd-ef94-4106-9c6d-a1e529bbe0a5" />
<img width="790" height="475" alt="image" src="https://github.com/user-attachments/assets/857e1acb-ddbc-4752-88c1-4edd07e6e402" />


Non-Emergency Vehicle Prediction
<img width="865" height="628" alt="image" src="https://github.com/user-attachments/assets/ff550d74-5f54-4734-a40b-a12679610b98" />

<img width="419" height="302" alt="image" src="https://github.com/user-attachments/assets/4cc04fc2-b8e9-4329-ba2e-26def7ee6eb3" />


### Future Improvements
Potential enhancements include:

Support for additional vehicle categories.
Transfer learning using MobileNetV2, EfficientNet, or ResNet.
Model explainability using Grad-CAM.
Deployment to Streamlit Community Cloud.
Confidence thresholding and uncertainty warnings.
Mobile-friendly user interface.

### Conclusion
This project successfully developed an end-to-end image classification system capable of distinguishing emergency vehicles from non-emergency vehicles using a Convolutional Neural Network.

The project progressed through data preprocessing, model development, hyperparameter tuning, evaluation, model persistence, application development, and deployment preparation. The final solution demonstrates how deep learning models can be transformed into practical applications through the integration of machine learning and web technologies.

