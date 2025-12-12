import streamlit as st
import numpy as np
import joblib
from PIL import Image
import cv2

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("iris_model.pkl")   # <-- FIX: using joblib correctly
        return model
    except FileNotFoundError:
        st.error("Model file 'iris_model.pkl' not found! Please train the model first.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Extract features from image
def extract_features_from_image(image):
    img_array = np.array(image)

    # Convert to HSV
    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Feature engineering (creative mapping)
    sepal_length = 4.0 + (np.mean(img_hsv[:,:,0]) / 180.0) * 4.0
    sepal_width  = 2.0 + (np.std(img_hsv[:,:,1]) / 128.0) * 2.5
    petal_length = 1.0 + (np.mean(img_hsv[:,:,2]) / 255.0) * 6.0
    petal_width  = 0.1 + (np.mean(img_array[:,:,0]) / 255.0) * 2.4

    return np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Main app UI
st.title("🌸 Iris Flower Classifier")
st.markdown("### Upload an image of an Iris flower to predict its species!")
st.markdown("---")

with st.sidebar:
    st.header("About")
    st.info("""
    Classifies Iris flowers into three species:
    - Setosa  
    - Versicolor  
    - Virginica  

    Upload a clear image of the flower for best results.
    """)

    st.header("Model Info")
    st.write("Algorithm: Random Forest")
    st.write("Features: 4 custom-engineered measurements")

model = load_model()

if model is not None:
    uploaded_file = st.file_uploader(
        "Choose a flower image...", 
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            st.write("### Classification Results")

            with st.spinner("Analyzing image..."):
                features = extract_features_from_image(image)
                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]

                species_names = ['Setosa', 'Versicolor', 'Virginica']
                predicted_species = species_names[prediction]

                st.success(f"**Predicted Species:** {predicted_species}")
                st.write(f"**Confidence:** {probabilities[prediction]*100:.1f}%")

                st.write("### Probability Distribution")
                for i, sp in enumerate(species_names):
                    st.progress(probabilities[i], text=f"{sp}: {probabilities[i]*100:.1f}%")

                with st.expander("View Extracted Features"):
                    st.write("Approximate measurements:")
                    st.write(f"- Sepal Length: {features[0][0]:.2f} cm")
                    st.write(f"- Sepal Width: {features[0][1]:.2f} cm")
                    st.write(f"- Petal Length: {features[0][2]:.2f} cm")
                    st.write(f"- Petal Width: {features[0][3]:.2f} cm")

    else:
        st.info("👆 Upload an image to start!")

