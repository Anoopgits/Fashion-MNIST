import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load the model
model = load_model("Fashion-mnist.h5")

# Title and description
st.title("Fashion-MNIST App")
st.write("Upload an image of clothing item to predict its category.")

# File uploader
file_uploaded = st.file_uploader("Upload the image", type=['jpg', 'png', 'jpeg'])

if file_uploaded is not None:
    # Read image
    image = Image.open(file_uploaded).convert("L")  # convert to grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize and preprocess
    image = np.array(image)  # convert to numpy
    image_resized = cv2.resize(image, (28, 28))
    image_resized = image_resized / 255.0  # normalize
    image_reshape = np.reshape(image_resized, (1, 28, 28, 1))

    # Prediction
    prediction = model.predict(image_reshape)
    prediction_label = np.argmax(prediction)

    # Fashion MNIST labels
    labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    st.success(f"The model predicts this clothing item is: **{labels[prediction_label]}**")
