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

# 1. Imports:
Libraries like cv2, PIL, tensorflow, numpy, pandas, etc., are imported for image processing, neural network operations, and result visualization.
# 2. train_model() function:
Image loading: It loads and preprocesses MRI images stored in the "datasets/no/" and "datasets/yes/" directories.
Data preprocessing: Images are resized to 64x64 pixels, converted to arrays, and normalized. Labels (0 for no tumor, 1 for tumor) are created accordingly.
Train-test split: The dataset is split into training (80%) and testing (20%) sets.
Model architecture: A sequential CNN model is created with:
# 3 convolutional layers (Conv2D) with ReLU activation and MaxPooling layers.
A fully connected (Dense) layer followed by a dropout layer (to reduce overfitting).
A final Dense layer with a softmax activation function for binary classification.
Training: The model is trained for 10 epochs using categorical cross-entropy as the loss function and the Adam optimizer. The training and validation accuracy are stored.
Saving the model: The trained model and the training/testing data are saved.
# 3. test_model() function:
Image prediction: This function loads a new MRI image for prediction, preprocesses it (resize and normalize), and makes a prediction using the trained model.
Thresholding: The model outputs probabilities, and if the tumor probability exceeds 50%, it classifies the image as having a tumor.
Evaluation: It prints the test accuracy and loss for the model on the test dataset.
# 4. plot_confusion_matrix() function:
This function generates a confusion matrix for the test results and visualizes it using Seaborn's heatmap for a better understanding of model performance.
# 5. Main Execution Block:
Training: The model is trained using the train_model() function.
Accuracy Table: The training and validation accuracy for each epoch is printed in a tabular format.
Model evaluation: Test accuracy and loss are evaluated.
Model summary: Prints the architecture of the model.
Prediction on new image: The model is tested on a new image (pred/pred2.jpg) and the prediction is printed.
Confusion matrix & classification report: The confusion matrix and classification report (precision, recall, F1-score) for the test set are displayed.
