# Brain_Tumor_Detection

This Python script is used for binary classification of brain MRI images to detect tumors. It builds a CNN model with convolutional layers to extract features from images and uses dense layers to classify them into "No Tumor" or "Tumor" categories. The model is trained, evaluated, and tested on both the dataset and a specific image. Results like accuracy, confusion matrix, and classification report are generated to assess the performance.

![image](https://github.com/OM-TRIPATHI1513/Brain_Tumor_Detection/assets/90430815/c4f54a08-2314-4469-9811-e19dca882654)

![image](https://github.com/OM-TRIPATHI1513/Brain_Tumor_Detection/assets/90430815/12690a4a-e306-4180-a7e1-4765d903c6d1)
![image](https://github.com/OM-TRIPATHI1513/Brain_Tumor_Detection/assets/90430815/15e7d5fc-7294-415f-a25d-bed221fbcf0d)
![image](https://github.com/OM-TRIPATHI1513/Brain_Tumor_Detection/assets/90430815/f46df0cf-711b-42af-961d-7dacb36d0282)

![image](https://github.com/OM-TRIPATHI1513/Brain_Tumor_Detection/assets/90430815/868a7372-d9e4-4d86-b2af-e610284ea17b)

![12-accuracy](https://github.com/OM-TRIPATHI1513/Brain_Tumor_Detection/assets/90430815/549242d2-a2fb-4371-8232-be8e644d97fb)

![12-loss](https://github.com/OM-TRIPATHI1513/Brain_Tumor_Detection/assets/90430815/0e1c2e59-1a64-481a-bdcb-f4bcda2f8019)

![20-accuracy](https://github.com/OM-TRIPATHI1513/Brain_Tumor_Detection/assets/90430815/3be97279-62a9-418d-bad8-ab967d53c9a3)

![20-loss](https://github.com/OM-TRIPATHI1513/Brain_Tumor_Detection/assets/90430815/46234f25-811c-497d-99d4-8c08fb1aa4a5)


# Brain Tumor Detection Using CNN

This project utilizes a Convolutional Neural Network (CNN) model to classify brain MRI images into two categories: **No Tumor** and **Tumor**. The dataset comprises MRI images of brain scans, and the CNN model is trained to identify the presence of a tumor.

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)
- [Code Breakdown](#code-breakdown)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Testing the Model](#testing-the-model)
- [Results and Evaluation](#results-and-evaluation)
- [Confusion Matrix & Classification Report](#confusion-matrix--classification-report)

## Project Overview

The goal of this project is to create a model that can classify MRI brain scan images into two categories:
- **No Tumor**
- **Tumor**

We achieve this using a CNN built with Keras and TensorFlow. The project involves image preprocessing, model training, evaluation, and testing on a new MRI image.

## Directory Structure

```plaintext
.
├── datasets/
│   ├── no/          # Directory containing MRI images with no tumor
│   └── yes/         # Directory containing MRI images with tumors
├── pred/
│   └── pred2.jpg    # Sample image for testing the model's prediction
├── my_model.keras   # Saved model after training
├── x_train.pkl      # Pickle file for training data (features)
├── y_train.pkl      # Pickle file for training labels
├── x_test.pkl       # Pickle file for testing data (features)
├── y_test.pkl       # Pickle file for testing labels
└── README.md        # Project documentation
```

## Dependencies

To run this project, ensure the following Python libraries are installed:

- TensorFlow
- Keras
- OpenCV
- PIL (Python Imaging Library)
- NumPy
- Matplotlib
- Seaborn
- Pandas
- Scikit-learn

Install the dependencies using the following command:
```bash
pip install tensorflow keras opencv-python pillow numpy matplotlib seaborn pandas scikit-learn
```

## Code Breakdown

### 1. **Imports:**
   The following libraries are used:
   - **cv2** and **PIL** for image processing.
   - **tensorflow** and **keras** for creating and training the CNN model.
   - **numpy** and **pandas** for data manipulation.
   - **seaborn** and **matplotlib** for visualizing the results.

### 2. **`train_model()` function:**

   - **Image Loading**: 
     Loads and preprocesses MRI images from the directories `datasets/no/` (no tumor) and `datasets/yes/` (tumor).

   - **Data Preprocessing**:
     - Images are resized to 64x64 pixels, converted to arrays, and normalized.
     - Labels are assigned (`0` for "No Tumor" and `1` for "Tumor").
   
   - **Train-Test Split**:
     The dataset is split into 80% for training and 20% for testing.

   - **Model Architecture**:
     A CNN model with the following layers:
     - **3 Convolutional layers**: Each with ReLU activation and MaxPooling for down-sampling.
     - **1 Fully Connected (Dense) layer**: Followed by a Dropout layer to reduce overfitting.
     - **Final Dense layer**: A softmax activation function for binary classification.

   - **Training**:
     The model is trained for 10 epochs using categorical cross-entropy as the loss function and Adam optimizer. Both training and validation accuracy are stored.

   - **Model Saving**:
     The trained model and the training/testing datasets are saved for later use.

### 3. **`test_model()` function:**

   - **Image Prediction**:
     This function loads a new MRI image from the `pred/` folder, preprocesses it (resize and normalize), and makes a prediction using the trained model.

   - **Thresholding**:
     The model outputs a probability of tumor presence. If the tumor probability exceeds 50%, the image is classified as "Tumor"; otherwise, "No Tumor".

   - **Evaluation**:
     Prints the model's test accuracy and loss on the test dataset.

### 4. **`plot_confusion_matrix()` function:**

   - **Confusion Matrix**:
     Generates and displays a confusion matrix using Seaborn’s heatmap. The confusion matrix helps visualize the model's performance by showing the number of correct and incorrect predictions for each class.

### 5. **Main Execution Block:**

   - **Training**:
     Calls the `train_model()` function to train the model.
   
   - **Accuracy Table**:
     Displays the training and validation accuracy for each epoch in a tabular format.

   - **Model Evaluation**:
     Test accuracy and loss are calculated after training.

   - **Model Summary**:
     Prints the architecture of the trained CNN model.

   - **Prediction on New Image**:
     The model is tested on a new image from the `pred/pred2.jpg` file, and the prediction (tumor or no tumor) is printed.

   - **Confusion Matrix & Classification Report**:
     After testing, a confusion matrix and classification report (precision, recall, F1-score) for the test set are displayed.

## Usage

### 1. Clone the repository:

```bash
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection
```

### 2. Place your MRI images in the `datasets/` folder under `no/` and `yes/` directories.

### 3. Run the Python script to train and test the model:

```bash
python script_name.py
```

## Training the Model

To train the CNN model, execute the following command:

```bash
python script_name.py
```

The model will load and preprocess images, split the data into training and testing sets, train the CNN model, and save the trained model.

## Testing the Model

After training, the model can be tested on new MRI images:
1. Place the new image in the `pred/` folder.
2. Run the script to get the prediction (tumor or no tumor).

Example output:
```bash
The probability of tumor presence is: 85.67%
Tumor presence: Yes

Test accuracy: 0.9234
Test loss: 0.2178
```

## Results and Evaluation

- **Training Accuracy**: The accuracy of the model on the training set after each epoch.
- **Validation Accuracy**: The accuracy of the model on the validation set after each epoch.
- **Test Accuracy & Loss**: The model's final accuracy and loss on the test dataset.

### Confusion Matrix & Classification Report

The confusion matrix and classification report are displayed to assess the model's performance. They show the number of true/false positives and negatives, as well as precision, recall, and F1-scores for both classes ("No Tumor" and "Tumor").

## Contributing

If you'd like to contribute, feel free to fork the repository and submit a pull request. For any issues, please open an issue on GitHub.

## License

This project is licensed under the MIT License.
