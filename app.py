import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("satellite_classifier.h5")

# Define class labels
categories = ["Cloudy", "Desert", "Green Area", "Water"]

# Streamlit UI
st.title("🌍 Satellite Image Classification")
st.write("Upload a satellite image, and the model will classify it as *Cloudy, Desert, Green Area, or Water*.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image_pil = Image.open(uploaded_file).convert("RGB")  # Ensure RGB format
    st.image(image_pil, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img = image_pil.resize((128, 128))  # Resize to match model input
    img_array = image.img_to_array(img)  # Convert to NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    # Debugging: Print shape
    st.write(f"Image Shape: {img_array.shape}")
    st.write(f"Expected Model Input Shape: {model.input_shape}")

    try:
        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions) * 100

        # Debugging: Print Raw Predictions
        st.write("🔹 Raw Model Output:", predictions)

        # Check if all predictions are favoring "Cloudy"
        if np.all(predictions < 0.5):  # If all classes have low confidence
            st.warning("⚠️ Model confidence is low. The model might not be well-trained.")
        
        # Show the result
        st.write(f"### 🛰 Predicted Class: *{categories[predicted_class]}*")
        st.write(f"### 🔍 Confidence: *{confidence:.2f}%*")

        # Show confidence scores for all classes
        st.write("#### Class Probabilities:")
        for i, category in enumerate(categories):
            st.write(f"🔹 {category}: {predictions[0][i]*100:.2f}%")
    
    except Exception as e:
        st.error(f"🚨 Prediction Error: {e}")
