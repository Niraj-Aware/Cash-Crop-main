#!/usr/bin/env python

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# Load pre-trained model
model = tf.keras.models.load_model('v3_pred_cott_dis.h5')

# Define labels for prediction output
labels = {
    0: 'Alternaria Alternata',
    1: 'Anthracnose',
    2: 'Bacterial Blight',
    3: 'Corynespora Leaf Fall',
    4: 'Healthy',
    5: 'Grey Mildew'
}

# Define function to preprocess input image
def preprocess_image(image):
    # Resize image
    image = image.resize((150,150))
    # Convert image to numpy array
    image = np.array(image)
    # Scale pixel values to range [0, 1]
    image = image / 150
    # Expand dimensions to create batch of size 1
    image = np.expand_dims(image, axis=0)
    return image

# Define function to make prediction on input image
def predict(image):
    # Preprocess input image
    image = preprocess_image(image)
    # Make prediction using pre-trained model
    prediction = model.predict(image)
    # Convert prediction from probabilities to label
    label = labels[np.argmax(prediction)]
    # Return label and confidence score
    return label, prediction[0][np.argmax(prediction)]

# Define Streamlit app
def main():
    # Set app title
    st.title('Cotton Plant Disease Detection [BETA]')
    # Set app description
    st.write('This app helps you to detect the type of disease in a cotton plant.')
    st.write('NOTE- This model only works on Cotton Plant. (Its under development, which predicts the exact disease of plant)')
    # Add file uploader for input image
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    submit = st.button('Predict')
    # On predict button click
    if submit:
        if uploaded_file is not None:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            # Displaying the image
            st.image(opencv_image, channels="BGR")
            # Resizing the image
            opencv_image = cv2.resize(opencv_image, (150, 150))
            # Convert image to 4 Dimension
            opencv_image = np.expand_dims(opencv_image, axis=0)
            # Make Prediction
            label, score = predict(Image.fromarray(cv2.cvtColor(opencv_image[0], cv2.COLOR_BGR2RGB)))
            st.write('Prediction: {} (confidence score: {:.2%})'.format(label, score))
            if label != 'Healthy':
                st.write("The cotton plant is infected with {}.".format(label))
                # Provide information about the disease
                if label == 'Alternaria Alternata':
                    st.write('Treatment options include removing infected plant parts and using fungicides.')
                elif label == 'Anthracnose':
                    st.write('Treatment options include removing infected plant parts and using fungicides.')
                elif label == 'Bacterial Blight':
                    st.write('Treatment options include removing infected plant parts and using bactericides.')
                elif label == 'Corynespora Leaf Fall':
                    st.write('Treatment options include removing infected plant parts and using fungicides.')
                else:
                    st.write('Treatment options include removing infected plant parts and using fungicides.')
            else:
                st.write("The cotton plant is healthy. No treatment is needed.")

if __name__ == '__main__':
    main()
