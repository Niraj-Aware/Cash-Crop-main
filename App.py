import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('path_to_your_model.h5')  # Change this to your model path
    return model

model = load_model()

# Function to preprocess image
def preprocess_image(image):
    # Preprocess the image here (resize, normalize, etc.)
    # Example: 
    image = image.resize((224, 224))  # Example resize
    image = np.array(image) / 255.0  # Example normalization
    return image

# Function to make prediction
def predict(image):
    processed_image = preprocess_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit app
def main():
    st.title('Cotton Plant Health Prediction')

    # File uploader
    uploaded_file = st.file_uploader("Upload an image of a cotton plant", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make prediction
        prediction = predict(image)
        if prediction[0][0] > 0.5:  # Example threshold for classifying as healthy
            st.write('Prediction: Healthy')
        else:
            st.write('Prediction: Diseased')

if __name__ == "__main__":
    main()
