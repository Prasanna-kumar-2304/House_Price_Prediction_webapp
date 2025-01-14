import streamlit as st
import pandas as pd
from joblib import load
from streamlit_folium import st_folium
import folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

@st.cache_resource
def load_model():
    model = load("models/model1.pkl")
    return model

model = load_model()

st.title("House Price Prediction with Pre-Trained Model")

feature_columns = ["SQUARE_FT", "BHK_NO.", "City_Tier", "READY_TO_MOVE", "RERA", "RESALE", "UNDER_CONSTRUCTION"]

@st.cache_data
def get_geolocation(lat, lon):
    geolocator = Nominatim(user_agent="geoapi")
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True, timeout=10)
        if location:
            address = location.raw.get('address', {})
            city = address.get('city', address.get('town', address.get('village', 'Unknown')))
            return city
    except GeocoderTimedOut:
        st.error("Geocoder request timed out. Please try again.")
    return 'Unknown'

st.write("### Select a Location on the Map")

m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

marker = folium.Marker([20.5937, 78.9629], draggable=True)
marker.add_to(m)

map_data = st_folium(m, width=700, height=500)

if map_data and "last_object_clicked" in map_data and map_data["last_object_clicked"] is not None:
    lat = map_data["last_object_clicked"]["lat"]
    lon = map_data["last_object_clicked"]["lng"]

    selected_city = get_geolocation(lat, lon)
    st.write(f"### Selected City: {selected_city}")

    area = st.number_input("Area (in sqft):", min_value=100, max_value=10000, step=10)
    bhk_no = st.number_input("Number of Bedrooms:", min_value=1, max_value=10, step=1)
    ready_to_move = st.selectbox("Ready to Move:", [0, 1])

    def mapping_city(city):
        tier_1 = ["Ahmedabad", "Bengaluru", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai", "Pune"]
        tier_2 = [
            "Agra", "Ajmer", "Aligarh", "Amravati", "Amritsar", "Aurangabad", "Bareilly", "Bhopal", "Chandigarh", 
            "Coimbatore", "Dehradun", "Lucknow", "Madurai", "Nagpur", "Patna", "Ranchi", "Thiruvananthapuram", "Varanasi"
        ]

        if city in tier_1:
            return 0 
        elif city in tier_2:
            return 1  
        else:
            return 2  


    city_tier = mapping_city(selected_city)

    input_data = {
        "SQUARE_FT": area,
        "BHK_NO.": bhk_no,
        "City_Tier": city_tier,  
        "READY_TO_MOVE": ready_to_move,
        "RERA": 0,  
        "RESALE": 0, 
        "UNDER_CONSTRUCTION": 0
    }

    input_df = pd.DataFrame([input_data])

    if st.button("Predict Price"):
        if area > 0 and bhk_no > 0:
            try:
                
                prediction = model.predict(input_df[feature_columns])
                price_in_lakhs = prediction[0] * 100000  #
                st.write(f"### Predicted Price: ₹{price_in_lakhs:,.2f} INR")
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
        else:
            st.error("Please provide valid inputs for all fields.")
else:
    st.write("Please select a location on the map by clicking on it.")
