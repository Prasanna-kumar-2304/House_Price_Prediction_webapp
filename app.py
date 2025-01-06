import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from joblib import load
from streamlit_folium import st_folium
import folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import matplotlib.pyplot as plt  # Corrected import
import seaborn as sns

# Load Pre-Trained Model
@st.cache_resource
def load_model():
    model = load("models/model1.pkl")  # Path to your saved model
    return model

model = load_model()

# Title
st.title("House Price Prediction with Pre-Trained Model")

# List of states used during training
state_list = ["Tamil Nadu", "Kerala", "Karnataka", "Maharashtra"]  # Adjust the list of states

# Full list of required features (must match the training dataset)
required_features = [
    "AREA", "BHK_NO.", "READY_TO_MOVE", 
    "STATE_Tamil Nadu", "STATE_Kerala", "STATE_Karnataka", "STATE_Maharashtra",
    "BATHROOMS", "FLOORS", "AGE", "PARKING", "ADDITIONAL_FEATURE1", "ADDITIONAL_FEATURE2"  # Replace with actual features
]

# Cache geocoding results
@st.cache_data
def get_geolocation(lat, lon):
    geolocator = Nominatim(user_agent="geoapi")
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True, timeout=10)
        if location:
            return location.raw['address'].get('state', 'Unknown')
    except GeocoderTimedOut:
        st.error("Geocoder request timed out. Please try again.")
    return 'Unknown'

# Interactive Map for Location Selection
st.write("### Select a Location on the Map")

# Set initial map location (e.g., central India)
m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

# Add marker to the map
marker = folium.Marker([20.5937, 78.9629], draggable=True)
marker.add_to(m)

# Render map in Streamlit
map_data = st_folium(m, width=700, height=500)

if map_data and "last_object_clicked" in map_data and map_data["last_object_clicked"] is not None:
    lat = map_data["last_object_clicked"]["lat"]
    lon = map_data["last_object_clicked"]["lng"]

    # Get state with caching
    selected_state = get_geolocation(lat, lon)
    st.write(f"### Selected State: {selected_state}")

    # User Inputs
    area = st.number_input("Area (in sqft):", min_value=100, max_value=10000, step=10)
    bhk_no = st.number_input("Number of Bedrooms:", min_value=1, max_value=10, step=1)
    ready_to_move = st.selectbox("Ready to Move:", [0, 1])  # 0 = No, 1 = Yes

    # Encode state dynamically
    encoded_state = [1 if state == selected_state else 0 for state in state_list]
    if sum(encoded_state) == 0:
        st.warning(f"State '{selected_state}' not recognized. Defaulting to 'Unknown'.")
        encoded_state = [0] * len(state_list)

    # Prepare input data for prediction (using the new model)
    input_data = {
        "SQUARE_FT": area,
        "BHK_NO.": bhk_no,
        "City_Tier": 1,  # Default value, should map dynamically
        "READY_TO_MOVE": ready_to_move,  # User input
        "RERA": 0,  # Default value
        "RESALE": 0,  # Default value
        "UNDER_CONSTRUCTION": 0  # Default value
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # List of feature columns used in the model
    feature_columns = ["SQUARE_FT", "BHK_NO.", "City_Tier", "READY_TO_MOVE", "RERA", "RESALE", "UNDER_CONSTRUCTION"]

    # Mapping cities to tiers manually
    def mapping_city(city):
        tier_1 = ["Ahmedabad", "Bengaluru", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai", "Pune"]
        tier_2 = ["Agra", "Ajmer", "Aligarh", "Amravati", "Amritsar", "Asansol", "Aurangabad", "Bareilly", "Belgaum", "Bhavnagar", "Bhiwandi", "Bhopal", "Bhubaneswar", "Bikaner", "Bilaspur", "Bokaro Steel City", "Chandigarh", "Coimbatore", "Cuttack", "Dehradun", "Dhanbad", "Bhilai", "Durgapur", "Erode", "Faridabad", "Firozabad", "Ghaziabad", "Gorakhpur", "Gulbarga", "Guntur", "Gwalior", "Gurugram", "Guwahati", "Hamirpur", "Hubli–Dharwad", "Indore", "Jabalpur", "Jaipur", "Jalandhar", "Jalgaon", "Jammu", "Jamnagar", "Jamshedpur", "Jhansi", "Jodhpur", "Navi Mumbai", "Kakinada", "Kannur", "Kanpur", "Karnal", "Kochi", "Kolhapur", "Kollam", "Kozhikode", "Kurnool", "Ludhiana", "Lucknow", "Madurai", "Malappuram", "Mathura", "Mangalore", "Meerut", "Moradabad", "Mysore", "Nagpur", "Nanded", "Nashik", "Nellore", "Navi Mumbai", "Noid", "Patna", "Puducherry", "Purulia", "Prayagraj", "Raipur", "Rajkot", "Rajamahendravaram", "Ranchi", "Rourkela", "Ratlam", "Salem", "Sangli", "Shimla", "Siliguri", "Solapur", "Srinagar", "Surat", "Thanjavur", "Thiruvananthapuram", "Thrissur", "Tiruchirappalli", "Tirunelveli", "Tiruvannamalai", "Ujjain", "Vijayapura", "Vadodara", "Varanasi", "Vasai-Virar City", "Vijayawada", "Visakhapatnam", "Vellore", "Warangal"]

        if city in tier_1:
            return 0
        elif city in tier_2:
            return 1
        else:
            return 2

    # Get the city tier based on input location (you can enhance this with more city data)
    city_tier = mapping_city(selected_state)

    # Update input with correct city tier
    input_data["City_Tier"] = city_tier
    input_df = pd.DataFrame([input_data])

    # Prediction Button
    if st.button("Predict Price"):
        if area > 0 and bhk_no > 0:  # Check for valid inputs
            try:
                # Ensure correct feature order
                prediction = model.predict(input_df[feature_columns])  
                price_in_lakhs = prediction[0] * 100000  # Convert to INR
                st.write(f"### Predicted Price: ₹{price_in_lakhs:,.2f} INR")
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
        else:
            st.error("Please provide valid inputs for all fields.")
else:
    st.write("Please select a location on the map by clicking on it.")

