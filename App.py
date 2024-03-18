import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# Load pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('v3_pred_cott_dis.h5')  # Change this to your model path
    return model

model = load_model()

# Define Streamlit app
def main():
    # Set app title
    st.title('Cotton Plant Disease Detection')
    # Set app description
    st.write('This app helps you to detect whether a cotton plant is healthy or diseased.')
    st.write('NOTE- This model only works on Cotton Plant. (With appropriate Image)')
    # Add file uploader for input image
    plant_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    # If file uploaded, display it and make prediction
    if plant_image is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Resize the image
        opencv_image = cv2.resize(opencv_image, (256, 256))
        
        # Convert BGR to RGB
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        
        # Displaying the image
        st.image(opencv_image, channels="RGB", caption='Uploaded Image', use_column_width=True)
        
        # Make Prediction
        prediction = model.predict(np.expand_dims(opencv_image, axis=0))
        result = prediction[0]  # Assuming the model returns a single prediction
        
        # Display result
        st.write('Prediction:', result)

if __name__ == '__main__':
    main()
