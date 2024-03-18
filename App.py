import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Load pre-trained ResNet50 model
model = tf.keras.applications.ResNet50(weights='imagenet')

# Function to preprocess input image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.asarray(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# Function to classify the uploaded image
def classify_image(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    return decoded_predictions[0][1]

# Streamlit app
def main():
    st.title("Cotton Plant Disease Detection")
    st.write("Upload an image of a cotton plant to check if it's healthy or diseased.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Classify the image
        result = classify_image(image)
        st.write(f"Prediction: {result}")

# Run the app
if __name__ == "__main__":
    main()
