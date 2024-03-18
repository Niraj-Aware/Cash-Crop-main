import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('v3_pred_cott_dis.h5')  # Change this to your model path
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

# Define Streamlit app
def main():
    # Set app title
    st.title('Cotton Plant Disease Detection')
    # Set app description
    st.write('This app helps you to detect whether a cotton plant is healthy or diseased.')
    st.write('NOTE- This model only works on Cotton Plant. (With appropriate Image)')
    # Add file uploader for input image
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    # If file uploaded, display it and make prediction
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        
        # If image mode is not RGB, convert it to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Display image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        # Make prediction
        prediction = predict(image)
        # Display prediction
        st.write('Prediction:', prediction)

if __name__ == '__main__':
    main()
