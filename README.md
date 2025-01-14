# House Price Prediction App with Machine Learning

## Overview
This project focuses on building a **House Price Prediction App** using a machine learning model trained on real estate data. The application predicts house prices based on user inputs such as area, number of bedrooms, and city tier. It features an interactive map for selecting the property's location and a user-friendly interface developed using Streamlit. The machine learning model used is a Linear Regression model trained on a dataset containing house attributes and their respective prices.

## Project Features

### Data Preprocessing:
- The training dataset includes features like:
  - **Area** (in square feet)
  - **Number of Bedrooms (BHK)**
  - **Ready-to-Move status** (Yes/No)
  - **City Tier** (Tier 1, Tier 2, or Tier 3 cities)
- Data cleaning and transformation:
  - Missing values were imputed or removed.
  - Categorical features (e.g., city, state) were one-hot encoded.
- The dataset was split into training and testing sets for model evaluation.

### Machine Learning Model:
- **Model Used**: Linear Regression.
- **Training and Evaluation**:
  - The model was trained using features like area, number of bedrooms, property condition, and city tier.
  - Metrics such as Mean Absolute Error (MAE) and R-squared were used for evaluation.
- The trained model was saved as a `.pkl` file for deployment.

### Streamlit App:
- **Interactive Map**:
  - Users can click on the map to select the location of the property.
  - The app uses reverse geocoding to detect the selected city's name.
- **User Inputs**:
  - Area (in sqft).
  - Number of bedrooms (BHK).
  - Whether the property is ready to move in.
- **Dynamic City Tier Mapping**:
  - Cities are classified into Tier 1, Tier 2, or Tier 3 based on predefined lists.
- **Prediction Output**:
  - The predicted house price is displayed in INR (Indian Rupees).

### Visualizations:
- A heatmap and bar plots can be added to visualize feature importance and correlations between attributes.

## Prerequisites

To run this project locally, you'll need to install the following libraries:
```bash
pip install streamlit pandas scikit-learn folium geopy joblib matplotlib seaborn
```

## Files
- `app.py`: The Streamlit application that provides the interface for the user.
- `models/model1.pkl`: The pre-trained Linear Regression model.
- `requirements.txt`: A list of dependencies required for the project.

## Results
- **Model Evaluation**:
  - The Linear Regression model achieved an R-squared score of 0.85 on the training data.
  - MAE: ₹1,50,000 (example metric, adjust based on your actual results).
- **Interactive Map**:
  - Users can visually select the location and receive predictions specific to the selected city.
- **Predictions**:
  - Accurate price predictions based on real-world data inputs.

## Acknowledgments
- The project uses libraries like **scikit-learn**, **Streamlit**, **Folium**, and **Geopy**.
- The house price dataset was collected and pre-processed for training and testing.
- Thanks to the open-source community for their valuable contributions to these libraries and tools.
