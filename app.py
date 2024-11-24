import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from io import BytesIO

# Load the model
@st.cache_resource  # To cache the loaded model
def load_classification_model():
    model = load_model('models/leaf_classification_model.h5')
    return model

model = load_classification_model()

# Class labels mapping (ensure this matches your training data class_indices)
class_labels = {0: 'Angular Leaf Spot', 1: 'Bean Rust', 2: 'Healthy Leaf'}

# Streamlit app UI
st.title("Leaf Disease Classification")
st.write("Upload an image of a leaf, and the app will predict whether it is healthy or has a disease.")

# File uploader
uploaded_file = st.file_uploader("Choose a leaf image (.jpg format)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open the uploaded image using PIL
        img = Image.open(uploaded_file).convert('RGB')  # Ensure RGB formatstre
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image to match model input requirements
        img = img.resize((224, 224))  # Resize to 224x224 as expected by the model
        img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Perform prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)  # Get the index of the highest probability
        predicted_label = class_labels.get(predicted_class, "Unknown")

        # Display the prediction
        confidence = np.max(prediction) * 100
        st.write(f"Predicted Label: {predicted_label}, Confidence: {confidence:.2f}%")

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.write("Please upload an image to get a prediction.")
