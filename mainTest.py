import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('BrainTumor10EpochsCategorical.h5')

# Read and preprocess the image
image = cv2.imread(
    'C:\\Users\\omtri\\OneDrive\\Desktop\\BrainTumor Classification DL\\pred\\pred2.jpg')
img = Image.fromarray(image)
img = img.resize((64, 64))
img = np.array(img)

# Preprocess the image - assuming normalization to [0, 1] during training
input_img = img.astype('float32') / 255.0
input_img = np.expand_dims(input_img, axis=0)

# Make predictions
predicted_class_index = np.argmax(model.predict(input_img), axis=-1)
print(predicted_class_index)
