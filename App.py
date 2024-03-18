#!/usr/bin/env python

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Define function to preprocess input image
def preprocess_image(image):
    # Resize image
    image = image.resize((150,150))
    # Convert image to numpy array
    image = np.array(image)
    # Scale pixel values to range [0, 1]
    image = image / 255.0
    # Expand dimensions to create batch of size 1
    image = np.expand_dims(image, axis=0)
    return image

# Define Streamlit app
def main():
    # Set app title
    st.title('Cotton Plant Disease Detection')
    # Set app description
    st.write('This app helps you to detect whether a cotton plant is healthy or diseased.')
    st.write('NOTE- This model only works on Cotton Plant. (With appropriate Image)')
    
    # Load pre-trained model
    @st.cache(allow_output_mutation=True)
    def load_model():
        return tf.keras.models.load_model('v3_pred_cott_dis.h5')

    model = load_model()
    
    # Define labels for prediction output
    labels = ['diseased','healthy']
    
    # Add file uploader for input image
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    
    # If file uploaded, display it and make prediction
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        # Display image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        # Preprocess input image
        processed_image = preprocess_image(image)
        # Make prediction using pre-trained model
        prediction = model.predict(processed_image)
        # Convert prediction from probabilities to label
        label = labels[np.argmax(prediction)]
        confidence_score = prediction[0][np.argmax(prediction)]
        # Display prediction
        st.write(f"Prediction: {label} (confidence score: {confidence_score:.2%})")
        # Provide instructions based on prediction
        if label == 'diseased':
            st.write('Your cotton plant appears to be diseased. To prevent the spread of disease, you should remove the infected plant and treat the soil. You can also consult a local agricultural expert for advice on how to prevent future outbreaks of disease.')
        else:
            st.write('Your cotton plant appears to be healthy. To keep it healthy, make sure to provide adequate water and fertilize regularly. You should also control pests and prune and train the plant to promote healthy growth. Harvest at the right time to ensure the highest quality fiber.')

# Run Streamlit app
if __name__ == '__main__':
    main()
