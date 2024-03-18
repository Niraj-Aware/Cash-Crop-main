import streamlit as st
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import tensorflow as tf

# Function to preprocess input image
def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to load pre-trained model from GitHub
@st.cache(allow_output_mutation=True)
def load_model():
    model_url = 'https://github.com/username/repo/raw/main/v3_pred_cott_dis.h5'  # Replace with your GitHub URL
    response = requests.get(model_url)
    model = tf.keras.models.load_model(BytesIO(response.content))
    return model

# Function to make prediction
def predict(image, model):
    image = preprocess_image(image)
    prediction = model.predict(image)
    label = "diseased" if prediction[0][0] > 0.5 else "healthy"
    confidence_score = prediction[0][0] if label == "diseased" else 1 - prediction[0][0]
    return label, confidence_score

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

        # Load pre-trained model
        model = load_model()

        # Make prediction
        label, confidence_score = predict(image, model)

        # Display prediction
        st.write(f"Prediction: {label} (confidence score: {confidence_score:.2f})")
        if label == "diseased":
            st.write("Your cotton plant appears to be diseased.")
        else:
            st.write("Your cotton plant appears to be healthy.")

# Run the app
if __name__ == "__main__":
    main()
