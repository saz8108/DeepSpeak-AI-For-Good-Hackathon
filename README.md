# DeepSpeak-AI-For-Good-Hackathon

## Project Overview

This project aims to provide an AI-powered solution for grocery stores and supermarkets to manage food waste and optimize the ordering process for fresh produce. The solution consists of the following key features:

1. **Fruit Quality Grading**: Using computer vision, the solution classifies fruits and vegetables into three categories:
   - **Fresh**
   - **Slightly Damaged**
   - **Heavily Damaged**

   This feature helps supermarkets apply price discrimination based on the condition of the produce, thereby reducing food waste and improving revenue management.

2. **Demand Forecasting**: Using historical data on food waste and produce sales, the solution forecasts future demand for fruits and vegetables, helping supermarkets optimize inventory management and reduce food waste.

3. **Inventory Management**: Based on food waste levels, the system suggests dynamic pricing for products to reduce wastage and improve stock management.

The full solution is available on a **Streamlit web app** for easy interaction and testing of different features.

[View the live Streamlit demo](https://deepspeak-ai-for-good-hackathon-ct287k8aj3f4ujcunw8pgw.streamlit.app/)

## Dataset

The **Fruits Dataset** used for fruit quality grading and model training was sourced from Kaggle. This dataset is licensed under the **Apache 2.0 License**, and it contains images of different types of fruits and vegetables categorized into several classes.

You can access the dataset on Kaggle here:
- [Fruits Dataset on Kaggle](https://www.kaggle.com/datasets/shivamardeshna/fruits-dataset)

## Technologies Used

- **Streamlit**: For building the interactive web app interface.
- **TensorFlow/Keras**: For building and training the fruit quality grading model.
- **Matplotlib**: For visualizing insights and data trends.
- **Pandas**: For data manipulation and analysis.

## Results

The fruit quality grading model classifies produce into three categories:
- **Fresh**: Produce that is suitable for sale at regular prices.
- **Slightly Damaged**: Produce that can be sold at discounted prices.
- **Heavily Damaged**: Produce that may need to be discarded or sold at very low prices.

The model acheived an 87% accuracy on testing data based on a very small dataset meaning it would be even better without these constraints.

In the **demand forecasting** section, the model uses historical data (timestamped food waste data) to predict the required quantities of fruits and vegetables for future periods, optimizing inventory levels and helping to reduce waste. This will still be implemented in the future.


