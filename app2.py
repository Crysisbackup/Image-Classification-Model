
import streamlit as st 
import cv2 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('D:\image classification project\models\happysadmodel.h5')
img_path =  "D:\\image classification project\\data\\happy\\getty_478389113_970647970450091_99776.jpg"
# Load and preprocess the image using cv2
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
resize = cv2.resize(img, (256, 256))
normalized = resize / 255.0
input_arr = np.expand_dims(normalized, 0)

# Make prediction
yhat = model.predict(input_arr)

st.image(img)
# Determine prediction and accuracy
prediction_score = yhat[0][0]
if prediction_score > 0.5:
    predicted_class = "Sad"
    accuracy = prediction_score * 100
else:
    predicted_class = "Happy"
    accuracy = (1 - prediction_score) * 100

st.write(f"Predicted class is: {predicted_class}")
st.write(f"Prediction accuracy: {accuracy:.2f}%")

 