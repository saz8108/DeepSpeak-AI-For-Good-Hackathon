import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import io
import matplotlib.pyplot as plt

# Set page config (optional)
st.set_page_config(page_title="Fruit Ninja App", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", 
    ["Dashboard", "Quality Grading", "Demand Forecast", "Inventory Management"])

# --- Utility Functions ---

@st.cache_resource
def load_model():
    """
    Load the pre-trained quality grading model from the file FruitNinja.h5.
    (Ensure this file is in the same directory as this app.py.)
    """
    model = tf.keras.models.load_model("FruitNinja.h5")
    return model

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess an image for model prediction.
    Resize to target_size and scale pixel values to [0, 1].
    """
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    elif isinstance(image, Image.Image):
        pass
    else:
        image = Image.open(image)
    image = image.resize(target_size)
    image_array = np.array(image)
    # If the image has an alpha channel, remove it.
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]
    image_array = np.expand_dims(image_array, axis=0)  # add batch dimension
    image_array = image_array / 255.0  # scale pixel values
    return image_array

# --- Page: Dashboard ---
if page == "Dashboard":
    st.title("Dashboard")
    st.write("## Overview and Insights")
    st.write("This dashboard provides insights from the data. (Placeholder content.)")
    
    # Example: Create a sample line chart with random data
    df = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'Fresh': np.random.randint(50, 100, size=10),
        'Slightly Damaged': np.random.randint(10, 50, size=10),
        'Heavily Damaged': np.random.randint(0, 10, size=10)
    })
    st.line_chart(df.set_index('Date'))
    st.write("Additional charts and insights can be added here.")

# --- Page: Quality Grading ---
elif page == "Quality Grading":
    st.title("Quality Grading")
    st.write("Upload an image of a banana to predict its quality.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Processing image...")
        
        # Preprocess image for model input
        processed_image = preprocess_image(image)
        
        # Load the pre-trained model
        model = load_model()
        
        # Run the prediction
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        # Mapping prediction to quality class
        class_names = ['Fresh', 'Slightly Damaged', 'Heavily Damaged']
        result = class_names[predicted_class]
        
        st.write(f"*Predicted Quality:* {result}")
    else:
        st.info("Please upload an image file to get started.")

# --- Page: Demand Forecast ---
elif page == "Demand Forecast":
    st.title("Demand Forecast")
    st.write("This page will provide demand forecasting insights. (Placeholder content)")
    
    # Example: Sample forecast chart
    forecast_data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=15, freq='D'),
        'Forecasted Demand': np.random.randint(100, 200, size=15)
    })
    st.line_chart(forecast_data.set_index('Date'))
    st.write("Additional forecasting tools and data can be added here.")

# --- Page: Inventory Management ---
elif page == "Inventory Management":
    st.title("Inventory Management & Dynamic Pricing")
    st.write("This page will help manage inventory and provide pricing suggestions to reduce waste.")
    
    # Example: Display sample inventory data
    inventory = pd.DataFrame({
        'Item': ['Fresh Bananas', 'Slightly Damaged Bananas', 'Heavily Damaged Bananas'],
        'Quantity': [100, 50, 20]
    })
    st.table(inventory)
    
    st.write("### Dynamic Pricing Suggestions")
    st.write("Adjust the slider to simulate expected food waste and see pricing suggestions.")
    
    # Slider for expected food waste percentage
    waste_percentage = st.slider("Expected Food Waste (%)", 0, 100, 20)
    
    # Dummy dynamic pricing suggestions (using placeholder formulas)
    discount_fresh = round(waste_percentage * 0.1, 2)
    discount_slightly = round(waste_percentage * 0.15, 2)
    discount_heavily = round(waste_percentage * 0.2, 2)
    
    pricing = pd.DataFrame({
        'Item': ['Fresh Bananas', 'Slightly Damaged Bananas', 'Heavily Damaged Bananas'],
        'Suggested Discount (%)': [discount_fresh, discount_slightly, discount_heavily]
    })
    st.table(pricing)
    st.write("Further analytics and management tools can be integrated here.")
