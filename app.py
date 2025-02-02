import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import io
import h5py
with h5py.File("FruitNinja.h5", "r") as f:
    model = tf.keras.models.load_model(f)
# Load the quality grading model
##@st.cache_resource
#def load_model():
    #return tf.keras.models.load_model("FruitNinja.h5")

#model = load_model()

# Function to preprocess and predict fruit quality
def predict_quality(image):
    image = image.resize((224, 224))  # Adjust size as per model requirement
    img_array = np.array(image) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    classes = ["Fresh", "Slightly Damaged", "Heavily Damaged"]
    return classes[np.argmax(prediction)], prediction

# Sidebar navigation
st.sidebar.title("📊 Food Waste Management")
page = st.sidebar.radio("Go to:", ["Dashboard", "Quality Grading", "Demand Forecasting", "Inventory Management"])

# **Dashboard Page**
if page == "Dashboard":
    st.title("📊 Food Waste Insights Dashboard")
    st.write("🚀 Placeholder for food waste analytics.")
    
    # Simulated KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("🍎 Fruits Wasted Today", "23 kg", "-12% vs Yesterday")
    col2.metric("📦 Inventory Remaining", "520 kg", "Steady")
    col3.metric("📈 Demand Forecast Accuracy", "89%", "+5% vs Last Week")

    # Simulated Graph
    st.subheader("📉 Food Waste Trends Over Time")
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4, 5], [50, 40, 30, 25, 23], marker="o", linestyle="-", color="red")
    ax.set_title("Waste Reduction Progress")
    ax.set_xlabel("Days")
    ax.set_ylabel("Wasted (kg)")
    st.pyplot(fig)

# **Quality Grading Page**
elif page == "Quality Grading":
    st.title("🍏 AI-Powered Fruit Quality Grading")
    uploaded_file = st.file_uploader("Upload an image of fruit/vegetable", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Predict quality
        label, confidence = predict_quality(image)
        st.subheader(f"🍎 Quality: **{label}**")
        st.write(f"Confidence: {np.max(confidence) * 100:.2f}%")

# **Demand Forecasting Page**
elif page == "Demand Forecasting":
    st.title("📊 AI-Powered Demand Forecasting")
    st.write("🚀 Placeholder for AI demand predictions.")
    
    # Simulated User Input
    st.subheader("📅 Select Timeframe")
    timeframe = st.selectbox("Timeframe:", ["Next 7 Days", "Next 30 Days", "Next 3 Months"])
    
    # Simulated Forecast Graph
    st.subheader("📈 Predicted Demand Trend")
    fig, ax = plt.subplots()
    ax.plot(["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"], [100, 120, 140, 135, 150], marker="o", linestyle="-")
    ax.set_title(f"Projected Demand for {timeframe}")
    ax.set_ylabel("Units Sold")
    st.pyplot(fig)

# **Inventory Management Page**
elif page == "Inventory Management":
    st.title("📦 Smart Inventory & Dynamic Pricing")
    
    # Placeholder table
    st.write("📋 **Current Inventory & Dynamic Pricing**")
    inventory_data = pd.DataFrame({
        "Product": ["Apples", "Bananas", "Oranges", "Tomatoes"],
        "Stock (kg)": [50, 30, 40, 60],
        "Predicted Waste (kg)": [5, 10, 8, 12],
        "Suggested Price ($/kg)": [2.0, 1.5, 1.8, 2.2],
    })
    st.dataframe(inventory_data)
    
    # Dynamic Pricing Suggestion
    st.subheader("💰 AI-Powered Price Adjustments")
    st.write("🚀 Placeholder for real-time price recommendations based on waste levels.")
