import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from PIL import Image as PILImage
import io


# Function to load and preprocess the image
def load_and_preprocess_image(image_bytes, target_size=(300, 300)):
    img = PILImage.open(io.BytesIO(image_bytes))
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalization step
    return img_array

# Function to make predictions
def predict_image(model, image_bytes):
    img = load_and_preprocess_image(image_bytes)
    prediction = model.predict(img)
    return prediction[0][0]



# Load the model
model = tf.keras.models.load_model('horse_or_human_model.h5')

# Title and file upload
st.title('Image Classification App (Horse or Human)')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Display uploaded image and make prediction
if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    st.image(image_bytes, caption='Uploaded Image.', use_column_width=False, width=500)
    st.write("")
    st.write("Classifying...")

    # Make prediction
    prediction = predict_image(model, image_bytes)

    # Display prediction
    if prediction > 0.5:
        st.write("Prediction: Human")
    else:
        st.write("Prediction: Horse")
    
